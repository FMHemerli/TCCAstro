# Contrato de API — TCCAstro

Este documento fixa a interface pública do núcleo. Ele existe para que o agente que escreve os
testes possa fazê-lo **sem ler a implementação**, e para que implementação e testes sejam derivados
independentemente da mesma fonte.

A física é definida em [`integradores.md`](integradores.md). Este documento define apenas **forma**:
nomes, assinaturas, tipos e contratos de erro. Onde os dois divergirem, `integradores.md` prevalece
e a divergência deve ser reportada.

O contrato é vinculante para ambos os lados. Quem precisar mudá-lo deve reportar a necessidade, não
mudar unilateralmente.

## Layout

```
src/nbody/
  config.py             constantes e conjuntos de parâmetros
  state.py              State
  initial_conditions.py geradores de condição inicial
  observables.py        diagnósticos físicos
  backends/__init__.py  registro de backends
  integrators.py        os quatro integradores e o laço de integração
```

Pacote instalável em modo editável a partir de `pyproject.toml`, importável como `nbody`.

## `nbody.config`

Expõe como constantes de módulo, em maiúsculas, todos os nomes listados na Seção 9 de
`integradores.md`, com exatamente aqueles valores (`G`, `PARTICLE_MASS`, `N_PARTICLES`,
`SPHERE_RADIUS`, `SOFTENING`, `SEED`, `T_FF`, `DT_COLLAPSE`, `N_STEPS_COLLAPSE`, e as demais).

Expõe também os conjuntos de parâmetros das Seções 4.5 a 4.8 como objetos congelados
(`RUN_COLLAPSE`, `RUN_CONVERGENCE`, `RUN_BENCH`, `TWOBODY_KEPLER`, `TWOBODY_CIRC`, `TWOBODY_ECC`),
com atributos nomeados conforme o documento de física.

## `nbody.state.State`

Contêiner imutável do estado dinâmico.

```python
@dataclass(frozen=True)
class State:
    r: torch.Tensor   # (N, 3), posições em m
    v: torch.Tensor   # (N, 3), velocidades em m/s
    m: torch.Tensor   # (N,),  massas em kg

    @property
    def n(self) -> int: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...

    def to(self, *, dtype=None, device=None) -> "State": ...
```

`r`, `v` e `m` compartilham dtype e device. A construção valida formas e consistência e levanta
`ValueError` quando violadas.

## `nbody.initial_conditions`

```python
def cold_sphere(n=N_PARTICLES, radius=SPHERE_RADIUS, particle_mass=PARTICLE_MASS,
                seed=SEED, dtype=torch.float64, device="cpu") -> State
def two_body_kepler(a=KEPLER_A, e=KEPLER_E, particle_mass=PARTICLE_MASS,
                    dtype=torch.float64, device="cpu") -> State
def two_body_circular(separation=CIRC_SEPARATION, softening=SOFTENING,
                      particle_mass=PARTICLE_MASS, dtype=torch.float64, device="cpu") -> State
```

`cold_sphere` segue o algoritmo normativo da Seção 5.2 e é **reprodutível bit a bit** para a mesma
semente, o mesmo `n` e o mesmo dtype, independentemente do device. Velocidades nulas.

## `nbody.observables`

Todas as reduções acumulam em fp64 independentemente do dtype do estado, conforme a diretiva da
Seção 6.4. Todas retornam `float` ou tensores fp64 no host.

```python
def kinetic_energy(state) -> float
def potential_energy(state, softening=SOFTENING) -> float
def total_energy(state, softening=SOFTENING) -> float
def linear_momentum(state) -> torch.Tensor    # (3,) fp64
def angular_momentum(state) -> torch.Tensor   # (3,) fp64
def half_mass_radius(state) -> float          # relativo ao centro de massa
def center_of_mass(state) -> torch.Tensor     # (3,) fp64
```

Nenhuma destas funções é chamada de dentro de uma região cronometrada. Instrumentação e dinâmica são
caminhos separados.

## `nbody.backends`

```python
BACKEND_NAMES: tuple[str, ...] = ("python_pure", "torch_eager", "torch_compiled", "triton")

def get_backend(name: str) -> Backend
def available_backends(device) -> tuple[str, ...]
```

`Backend` é um protocolo:

```python
class Backend(Protocol):
    name: str
    def supports(self, device: torch.device) -> bool: ...
    def accelerations(self, r, m, softening=SOFTENING, *, tile_size=None) -> torch.Tensor: ...
    def fused_symplectic_euler_step(self, state, dt, softening=SOFTENING) -> State: ...
```

- `accelerations` recebe `r` de forma `(N, 3)` e `m` de forma `(N,)`, e devolve `(N, 3)` no mesmo
  dtype e device. É a primitiva sobre a qual todos os integradores operam.
- Deve aceitar `softening=0.0` corretamente, conforme o requisito da Seção 2.5.
- `tile_size=None` significa escolher automaticamente a partir do teto de memória de `config`.
  Qualquer `tile_size` válido deve produzir o mesmo resultado que qualquer outro dentro de
  `TOL-IMPL-SHORT` — a ordem de redução muda, o valor não.
- `fused_symplectic_euler_step` replica o kernel de 2019 com força e atualização de estado fundidas,
  existindo apenas para o benchmark comparativo. Não é usado pelos integradores.
- `supports(device)` é honesto: `python_pure` não suporta GPU; `triton` não suporta CPU.

Combinações válidas — os seis degraus da escada de comparação:

| degrau | backend | device |
|---|---|---|
| 1 | `python_pure` | `cpu` |
| 2 | `torch_eager` | `cpu` |
| 3 | `torch_compiled` | `cpu` |
| 4 | `torch_eager` | `cuda` |
| 5 | `torch_compiled` | `cuda` |
| 6 | `triton` | `cuda` |

**Nenhum backend faz fallback silencioso.** Um backend indisponível levanta `BackendUnavailable`
(exportada pelo pacote) nomeando o que falta. Um backend nunca troca de device, de precisão ou de
caminho de código sem que o chamador tenha pedido.

## `nbody.integrators`

```python
INTEGRATOR_NAMES: tuple[str, ...] = ("euler", "symplectic_euler", "velocity_verlet", "rk4")
FORCE_EVALS_PER_STEP: dict[str, int]   # k, conforme a tabela da Secao 3.5
FORCE_EVALS_STARTUP: dict[str, int]    # s, custo de borda (pre-condicao)

def integrate(state, *, integrator, backend, dt, n_steps,
              softening=SOFTENING, callback=None, callback_every=None) -> State
```

- `integrator` é um nome de `INTEGRATOR_NAMES`; `backend` é um `Backend` ou um nome.
- Implementa exatamente a sequência de operações da Seção 3 correspondente, incluindo o
  reaproveitamento de aceleração entre passos onde o documento o especifica.
- `callback(step_index, time, state)` é chamado a cada `callback_every` passos, para amostragem de
  observáveis. Chamar `callback` fora do laço cronometrado é responsabilidade de quem cronometra;
  `integrate` apenas o invoca.
- Nomes desconhecidos levantam `ValueError` listando os válidos.

O custo total de `M` passos é `n_force = M * FORCE_EVALS_PER_STEP[name] + FORCE_EVALS_STARTUP[name]`.
Apenas `velocity_verlet` tem custo de borda não nulo (`s = 1`), pela avaliação de pré-condição
`a^0 = A(r^0)` que a Seção 3.3 exige. A fórmula vale inclusive para `M = 0`. Nenhum gráfico contra
`n_force` pode ser produzido sem usar as duas tabelas — ver Seção 8.2 de `integradores.md`.

## Contratos de erro

Exportados pelo pacote: `BackendUnavailable`.

Não há `except Exception` nu em lugar nenhum. Nenhuma condição de falha é convertida em resultado
degradado silencioso. Uma indisponibilidade é sempre uma exceção nomeada, nunca um retorno plausível.

## Convenções

Código, identificadores e docstrings em inglês. Sem emojis. Nenhuma assinatura, coautoria ou
referência a autoria de IA em qualquer artefato.
