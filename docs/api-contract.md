# Contrato de API — TCCAstro

Este documento fixa a interface pública do núcleo. Ele existe para que o agente que escreve os
testes possa fazê-lo **sem ler a implementação**, e para que implementação e testes sejam derivados
independentemente da mesma fonte.

A física é definida em [`integradores.md`](integradores.md) e, para as extensões estocásticas
(espectro de massas, velocidades térmicas, colisões), em
[`simulacao-estocastica.md`](simulacao-estocastica.md). Este documento define apenas **forma**:
nomes, assinaturas, tipos e contratos de erro. Onde divergirem, os documentos de física prevalecem
e a divergência deve ser reportada.

O contrato é vinculante para ambos os lados. Quem precisar mudá-lo deve reportar a necessidade, não
mudar unilateralmente.

> ## EMENDA 2026-08-08 — extensões estocásticas
>
> Este documento descrevia apenas o núcleo suave e **não havia sido emendado** para nada do que veio
> depois. Esta revisão aplica as emendas decididas e justificadas na **Seção 9.1 de
> `simulacao-estocastica.md`**, que é onde elas devem ser lidas com a justificativa física:
>
> - a condição inicial `random_sphere` (Seções 2 e 3);
> - o módulo `nbody.populations` (amostragem de massas e velocidades);
> - o módulo `nbody.collisions` (Seção 4);
> - os parâmetros `collision=` e `collision_rng=` de `integrate()`, e a mudança do seu **tipo de
>   retorno**;
> - a convenção `m = 0` para slot morto (Seção 5);
> - os observáveis `n_live`, `mass_spectrum_summary`, `scales_from_state`;
> - a **armadilha normativa** de que `collisions.detect` interpreta `state.v` como velocidade de
>   **meio passo**.
>
> Tudo o que já estava aqui continua válido sem alteração. `cold_sphere` em particular está
> **intocada** e continua sendo a referência bit a bit de `INV-17`.

## Layout

```
src/nbody/
  config.py             constantes e conjuntos de parametros
  state.py              State
  initial_conditions.py geradores de condicao inicial
  populations.py        amostragem de massas e velocidades (estocastico)
  collisions.py         deteccao, pareamento e resolucao de colisoes
  observables.py        diagnosticos fisicos
  backends/__init__.py  registro de backends
  integrators.py        os quatro integradores e o laco de integracao
```

Pacote instalável em modo editável a partir de `pyproject.toml`, importável como `nbody`.

## `nbody.config`

Expõe como constantes de módulo, em maiúsculas, todos os nomes listados na Seção 9 de
`integradores.md`, com exatamente aqueles valores (`G`, `PARTICLE_MASS`, `N_PARTICLES`,
`SPHERE_RADIUS`, `SOFTENING`, `SEED`, `T_FF`, `DT_COLLAPSE`, `N_STEPS_COLLAPSE`, e as demais).

**Acrescentado em 2026-08-08:** expõe também, com exatamente aqueles valores, todas as constantes da
**Seção 8 de `simulacao-estocastica.md`** — espectro de massas (`MASS_*`), velocidades (`Q_DEFAULT`,
`F_CUT_DEFAULT`, `VEL_*`), colisões (`CHI_DEFAULT`, `R_REF_DEFAULT`, `MAP_*`, `FRAG_*`,
`COLLISION_SEED`, `COLLISION_DRAWS_PER_EVENT`) e ensemble (`K_SEEDS`, `ENS_*`). Os símbolos
**removidos** por aquela seção (`DT_COLLISION`, `COH_VELOCITY_FACTOR`, `MAP_B`, `MAP_W`,
`MAP_S_CLAMP`, `ENS_DISPERSION_MAX`, `TOL_COURANT_MAX`, `TOL_REJECT_MAX`, `MASS_CAP_UNIFORM`,
`MASS_CAP_DEFAULT`) **não são símbolos deste projeto** e não podem reaparecer em `config`.

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

**Convenção `m = 0` — slot morto (emenda 2026-08-08; Seção 5 de `simulacao-estocastica.md`).**
A fusão é `2 -> 1`, mas as formas dos tensores **nunca mudam**: o slot liberado recebe `m = 0` em
vez de ser removido, para que `torch.compile` e o kernel Triton não recompilem e para que a cadeia
de reaproveitamento de aceleração do Verlet continue válida.

- **`State` ADMITE `m = 0` e `__post_init__` NÃO pode rejeitá-lo.** Uma validação `m > 0` quebra a
  fusão inteira. Isto é contrato, não tolerância.
- O slot morto fica em `r = r_fundido`, `v = v_fundido` (índice menor vence — Seção 4.9(0)).
- `m = 0` é **exatamente inerte**, bit a bit, em força, `U`, `K`, `P`, `L`, centro de massa e
  `half_mass_radius` (`INV-27`).
- **Única exceção, e é normativa:** com `softening == 0.0`, um slot morto **coincidente** com um
  corpo vivo produz `0 * inf = NaN` no campo de aceleração. Daí a recusa de `integrate()` descrita
  abaixo (`INV-28`). Não é um caso a mascarar no kernel; é um caso a recusar na borda.
- Massa **negativa** continua inválida. `m = 0` é ausência, não matéria exótica.

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

**Acrescentado em 2026-08-08** (Seções 2 e 3 de `simulacao-estocastica.md`):

```python
def random_sphere(n=N_PARTICLES, radius=SPHERE_RADIUS, seed=SEED,
                  mass_spectrum: dict | None = None,
                  virial_ratio: float = 0.0,
                  f_cut: float = F_CUT_DEFAULT,
                  mass_seed: int = MASS_SEED, vel_seed: int = VEL_SEED,
                  dtype=torch.float64, device="cpu") -> State
```

- **`cold_sphere` NÃO é substituída e NÃO muda.** Ela continua sendo a referência bit a bit de
  `INV-17`, e é contra ela que se verifica que `random_sphere` com `mass_spectrum=None` e
  `virial_ratio=0.0` reproduz o colapso frio existente.
- `mass_spectrum=None` significa **massas iguais** a `PARTICLE_MASS`. Um `dict` liga a lei de
  potência truncada da Seção 2 (`alpha`, `ratio`, e o condicionamento em `1 <= k <= 3` corpos
  massudos).
- `virial_ratio` é `Q = 2K/|U|`. `Q = 0.0` dá velocidades **exatamente** nulas — não "pequenas".
- `f_cut` é o teto de truncamento da maxwelliana, em unidades de `v_esc`. **Existe uma restrição de
  admissibilidade entre `Q` e `f_cut`**, verificada em tempo de execução:
  `Q <= Q_USABLE_COEFF * f_cut²`. O par `Q = 0.5, f_cut = 0.5` está **vetado** (Seção 3.4); os
  valores fixados são `Q = 0.25`, `f_cut = 0.5`.
- **Três fluxos aleatórios SEPARADOS e independentes**, e isto é contrato, não detalhe: `seed`
  (posições), `mass_seed` (massas), `vel_seed` (velocidades). Compartilhar um único fluxo correlaciona
  massa com posição e destrói a hipótese de amostragem independente sobre a qual `INV-11` a `INV-16`
  são escritos.

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

**Acrescentado em 2026-08-08** (Seção 9.1 de `simulacao-estocastica.md`):

```python
def n_live(state) -> int                      # numero de slots com m > 0
def mass_spectrum_summary(state) -> dict      # n_live, total/mean/min/max/std sobre os VIVOS
def scales_from_state(state, *, softening=SOFTENING, radius=SPHERE_RADIUS) -> dict
```

- As três ignoram slots mortos. `mass_spectrum_summary` levanta `ValueError` se **nenhum** corpo
  estiver vivo — degenerescência total é falha nomeada, não `nan` propagado.
- `scales_from_state` devolve as escalas da **massa realizada**, não das constantes de `config`:
  `total_mass`, `density`, `t_ff`, `t_conv`, `v_char`, `l_scale`, `u_min_bound`. Isto existe porque,
  com espectro de massas, `M_real != N * PARTICLE_MASS` (`CV = 9.24%`, `sd(t_ff)/t_ff = 4.62%`), e
  normalizar por `T_FF` de `config` introduz erro sistemático de vários por cento em `INV-15`,
  `INV-24` e `INV-31`.
- `u_min_bound` usa a forma com massas desiguais, `-G (M_real² - sum_i m_i²) / (2 eps)` — ver a
  emenda a `INV-10` em `integradores.md`.
- **`half_mass_radius` é a mediana de MASSA**, não de contagem (`INV-29`). Para massas iguais e `N`
  par o valor é **bit a bit** o da fórmula de contagem, de modo que nenhum valor publicado mudou.

Nenhuma destas funções é chamada de dentro de uma região cronometrada. Instrumentação e dinâmica são
caminhos separados.

## `nbody.populations`

**Módulo acrescentado em 2026-08-08.** Separa a geração estocástica (massas, velocidades) da
condição inicial geométrica. Física nas Seções 2 e 3 de `simulacao-estocastica.md`.

```python
def g_factor(alpha: float, ratio: float) -> float
def mass_min_from_mean(mean_mass: float, alpha: float, ratio: float) -> float
def power_law_cdf(m, m_min: float, m_max: float, alpha: float)
def power_law_inv_cdf(u, m_min: float, m_max: float, alpha: float)
def mass_big_threshold(n: int, m_min: float, m_max: float, alpha: float) -> float
def mass_k_weights(n: int, tail_prob: float) -> tuple[float, float, float]
def sample_masses(...)
def truncated_second_moment_factor(x: float) -> float
def solve_sigma(...)
def sample_velocities(...)
```

- `mass_min_from_mean` é **forma fechada**, `m_min = <m> / g(alpha, ratio)`. **Não há raiz a buscar**;
  uma implementação com iteração de Newton aqui está errada por construção, não por precisão.
- `sample_masses` implementa o algoritmo normativo da Seção 2.7, incluindo a **permutação uniforme
  dos slots massudos** — sem ela a construção deixa de ser a condicional exata e `INV-13` reprova.
  É o único invariante capaz de pegar a ausência.
- `solve_sigma` resolve a **equação implícita** no segundo momento truncado. Usar `<v²> = 3 sigma²`
  introduz erro sistemático de `3.9%`, três vezes o ruído amostral (Seção 3.3.2).
- `sample_velocities` devolve `lambda` (o fator de reescalonamento) **como retorno auxiliar**. Sem
  `lambda` exposto, `INV-16(c)` é intestável — este é o motivo de ele estar na interface pública, e
  não uma conveniência.

## `nbody.collisions`

**Módulo acrescentado em 2026-08-08.** Física na Seção 4 de `simulacao-estocastica.md`; as
assinaturas são fixadas normativamente na Seção 9.1.1 de lá e transcritas aqui.

```python
@dataclass(frozen=True)
class CollisionModel:
    r_ref: float = R_REF_DEFAULT
    m_bar: float = PARTICLE_MASS
    v_coh: float                      # kw-only, OBRIGATORIO, sem valor padrao
    seed: int = COLLISION_SEED        # kw-only

@dataclass(frozen=True)
class CollisionCandidates:
    i, j, t_star, rel_speed, contact_radius_sum: torch.Tensor   # 1-D, mesmo comprimento
    n: int                                                      # @property

@dataclass(frozen=True)
class AcceptedPairs:            # como CollisionCandidates, mais:
    f_reject: float             # DO PASSE

@dataclass(frozen=True)
class CollisionOutcome:
    state: State                # apos o passe, DRIFT COMPLETO
    n_elastic: int
    n_merge: int
    n_fragment: int
    delta_e_int: float          # fp64
    delta_l_spin: torch.Tensor  # (3,) fp64
    f_reject: float
    c_coll_max: float

def contact_radii(m, model: CollisionModel) -> torch.Tensor
def detect(state, dt: float, model: CollisionModel) -> CollisionCandidates
def pair_disjoint(candidates: CollisionCandidates) -> AcceptedPairs
def resolve(state, dt, model, accepted: AcceptedPairs, generator, softening) -> CollisionOutcome
def collision_pass(state, dt, model, generator, softening) -> CollisionOutcome
def v_coh_from_state(state, radius: float) -> float
def regime_probabilities(x, *, elastic_weight=MAP_ELASTIC_WEIGHT, x_clamp=MAP_X_CLAMP)
```

### ARMADILHA NORMATIVA de `detect` — `state.v` é a velocidade de MEIO PASSO

**`detect` recebe um `State`, mas interpreta `state.v` como `v^(n+1/2)`, a velocidade de meio passo
do Verlet, e não como `v^n`.**

```python
# CORRETO -- o chamador e' OBRIGADO a montar isto:
candidates = detect(State(r=r_n, v=v_half, m=m), dt, model)

# ERRADO -- roda, nao levanta nada, e detecta os pares errados:
candidates = detect(State(r=r_n, v=v_n, m=m), dt, model)
```

**`detect` não tem como verificar isso, e não vai avisar.** A varredura sobre `[0, dt]` é *exata*
para o drift do Verlet precisamente porque, nesse trecho, o movimento é retilíneo e uniforme com a
velocidade de meio passo (Seção 4.3). Passando `v^n`, a trajetória varrida não é a trajetória
percorrida: o detector devolve `t*` errados e pares errados — **e continua devolvendo resultados
plausíveis**, com contagens da ordem certa. Não há teste de sanidade barato que pegue isto a partir
da saída; o que pega é o contrato, e por isso ele está aqui e na docstring.

### Pontos normativos sobre `resolve`

1. **`resolve` é PURA quanto aos acumuladores.** Devolve os deltas **deste passe**
   (`delta_e_int`, `delta_l_spin`) e **não** mantém `E_int` nem `L_spin` correndo internamente. Os
   acumuladores de longo prazo pertencem ao laço de integração, em fp64. Estado mutável escondido
   num módulo é o que faz duas equipes divergirem, e tornaria `INV-23(a)` impossível de isolar.
2. **`resolve` completa o drift.** `CollisionOutcome.state` já tem os não participantes com
   `r += dt*v` e os participantes avançados `dt - t*` após o mapa. **O chamador não faz drift
   nenhum depois.**
3. **`softening` é parâmetro explícito**, como em `accelerations`, e não campo do modelo. `resolve`
   precisa dele para `E_grav`; `detect` não precisa e **não o recebe**.
4. **`CollisionModel.v_coh` é obrigatório e sem valor padrão**, deliberadamente: ele depende da massa
   **realizada** e do raio, e o chamador o computa com `v_coh_from_state`. Um valor padrão aqui seria
   uma sentinela silenciosamente errada sempre que `M_real != N * PARTICLE_MASS`. **Não acrescentar
   um padrão.**
5. **Exatamente `2` sorteios uniformes por evento aceito, para TODO canal** (`INV-32`), consumidos
   **antes** do desvio de canal. Sortear `u2` só dentro do ramo de fragmentação torna o consumo
   dependente do canal e a execução irreprodutível.

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
              softening=SOFTENING, callback=None, callback_every=None,
              collision: CollisionModel | None = None,
              collision_rng: np.random.Generator | None = None,
              ) -> State | tuple[State, CollisionRunStats]
```

- `integrator` é um nome de `INTEGRATOR_NAMES`; `backend` é um `Backend` ou um nome.
- Implementa exatamente a sequência de operações da Seção 3 correspondente, incluindo o
  reaproveitamento de aceleração entre passos onde o documento o especifica.
- `callback(step_index, time, state)` é chamado a cada `callback_every` passos, para amostragem de
  observáveis. Chamar `callback` fora do laço cronometrado é responsabilidade de quem cronometra;
  `integrate` apenas o invoca.
- Nomes desconhecidos levantam `ValueError` listando os válidos.

### Colisões em `integrate()` — acrescentado em 2026-08-08

**`collision` e `collision_rng` são somente por palavra-chave.**

- **`collision=None` (padrão) é o caminho existente, bit a bit.** `INV-30` exige que a trajetória
  seja **identicamente** a de antes de o parâmetro existir, para os quatro integradores. Nenhum
  reordenamento de operações no drift é permitido "de passagem".
- **Tipo de retorno.** Com `collision=None`, devolve `State`, como sempre. Com colisão ligada,
  devolve `tuple[State, CollisionRunStats]` — os acumuladores de longo prazo (`n_elastic`,
  `n_merge`, `n_fragment`, `delta_e_int`, `delta_l_spin`, `c_coll_max`, `f_reject_max`). Eles
  pertencem ao **laço**, e não a `collisions.py`, por decisão explícita (ponto 1 de `resolve`);
  `integrate` é onde o laço termina, logo é por aqui que eles saem.
- **Erros — nenhum vira resultado degradado:**

| condição | efeito |
|---|---|
| `collision is not None and softening == 0.0` | `ValueError` (`INV-28`). Slot morto coincidente com corpo vivo daria `0 * inf = NaN` no campo de aceleração e a simulação inteira viraria `NaN` |
| `collision is not None and integrator != "velocity_verlet"` | `ValueError`. A colisão é definida **dentro do drift** de Verlet (Seção 4.5) e não há definição de onde ela entraria nos outros três |
| `collision is None and collision_rng is not None` | `ValueError`. Um gerador entregue a um caminho que não o usa é erro do chamador, não configuração a ignorar em silêncio |

### `collision_rng` — o parâmetro, e o MOTIVO, que é o que impede a sua remoção

- **Omitido (`None`)**: `integrate` constrói `np.random.default_rng(collision.seed)` internamente e
  **reproduz bit a bit o comportamento anterior** para qualquer chamador que faça **uma única
  chamada por execução**. Nenhum resultado publicado muda.
- **Fornecido**: é o objeto que **o chamador mantém vivo entre chamadas**. `integrate` consome dele
  e **nunca o recria nem o re-semeia**.

**O motivo.** `INV-32` exige que o fluxo de colisão consuma exatamente `2` sorteios por evento
aceito, **de forma contínua ao longo da execução**, e `INV-19(c)` exige reprodutibilidade bit a bit
da execução inteira. A versão anterior construía o gerador **a cada chamada**, a partir de
`collision.seed`. Enquanto uma execução era uma única chamada, isso era indistinguível de um fluxo
contínuo. **Deixou de ser**: o visualizador em tempo real avança a simulação **em pedaços**, uma
chamada de `integrate` por quadro. Nesse padrão cada chamada reiniciava o fluxo do mesmo `seed`:

```
fluxo pretendido (uma chamada de M passos):   u_1 u_2 u_3 u_4 u_5 u_6 ...
fluxo realizado  (M chamadas de 1 passo):     u_1 u_2 | u_1 u_2 | u_1 u_2 | ...
```

**Consequências, todas físicas, e nenhuma delas visível como erro:** o sorteio de canal deixa de ser
independente entre quadros e passa a ser **periódico com o período do quadro**; as frações de canal
deixam de ser amostras do mapa de regime; e a mesma semente passa a dar resultados diferentes
conforme o tamanho do pedaço — exatamente o que `INV-19(c)` existe para impedir. **`INV-32`
continuava passando**, porque é enunciado **por passe**, de modo que a violação não aparecia em
nenhum teste da suíte.

> **Aviso a quem for simplificar.** `collision_rng` parece redundante: `CollisionModel` já tem
> `seed`, e o padrão reproduz o comportamento antigo. **Removê-lo — ou fazer `integrate` re-semear a
> partir de `model.seed` quando ele é fornecido — reintroduz exatamente o defeito acima, e
> reintroduz sem falhar teste nenhum da suíte atual.** Se a remoção for proposta, o ônus é
> apresentar o teste de equivalência fatiada/inteira passando sem o parâmetro. Ver a Seção 9.1.2 de
> `simulacao-estocastica.md`.

O custo total de `M` passos é `n_force = M * FORCE_EVALS_PER_STEP[name] + FORCE_EVALS_STARTUP[name]`.
Apenas `velocity_verlet` tem custo de borda não nulo (`s = 1`), pela avaliação de pré-condição
`a^0 = A(r^0)` que a Seção 3.3 exige. A fórmula vale inclusive para `M = 0`. Nenhum gráfico contra
`n_force` pode ser produzido sem usar as duas tabelas — ver Seção 8.2 de `integradores.md`.

## Contratos de erro

Exportados pelo pacote: `BackendUnavailable`.

Não há `except Exception` nu em lugar nenhum. Nenhuma condição de falha é convertida em resultado
degradado silencioso. Uma indisponibilidade é sempre uma exceção nomeada, nunca um retorno plausível.

**Acrescentado em 2026-08-08.** As três recusas de `integrate()` com colisão (tabela acima) são
`ValueError` e são **de borda**: a condição é verificada **antes** do primeiro passo, não descoberta
no meio da execução. `mass_spectrum_summary` sobre um estado sem corpos vivos também é `ValueError`.

**Reprodutibilidade é contrato de erro, não conveniência.** O fluxo de colisão consome **exatamente
dois** sorteios por evento aceito, na ordem de pareamento `(t*, i, j)`, para **todo** canal
(`INV-19(c)`, `INV-32`). Uma implementação que consuma número variável de sorteios não produz um
resultado "ligeiramente diferente": produz um resultado **não reprodutível**, que não é publicável.

## Convenções

Código, identificadores e docstrings em inglês. Sem emojis. Nenhuma assinatura, coautoria ou
referência a autoria de IA em qualquer artefato.
