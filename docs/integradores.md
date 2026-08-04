# Especificação física executável — TCCAstro

Documento normativo. Fixa o problema contínuo, a discretização, os parâmetros numéricos, os
invariantes testáveis e as tolerâncias do estudo de física e integradores do projeto TCCAstro.

**Status.** Vinculante. O agente de implementação escreve código a partir deste documento; o agente
de testes escreve testes a partir deste documento, sem ler a implementação. Toda constante,
tolerância e critério de aceitação necessários aos testes estão aqui. Se um teste precisar de um
número que não esteja neste documento, o documento está incompleto e deve ser corrigido — não o
teste.

**Convenção.** Texto em português; equações, identificadores e nomes de arquivo em inglês.
Unidades SI ao longo de todo o documento, mantendo as constantes do TCC de 2019.

**Rastreabilidade das afirmações.** Cada afirmação quantitativa está marcada como:
- **[T]** garantido teoricamente pelo método (demonstração algébrica);
- **[M]** medido nesta configuração específica (valor de referência obtido em fp64);
- **[A]** assumido, ainda não verificado.

Afirmações **[M]** valem para a configuração exata descrita na Seção 4. Não são propriedades gerais
do problema de N corpos.

---

## 1. Problema contínuo

### 1.1 Equações de movimento

Sistema de `N` partículas pontuais de massa `m_i`, posições `r_i ∈ R^3`, velocidades `v_i ∈ R^3`,
interagindo por gravitação newtoniana com regularização de Plummer de comprimento `eps`:

```
dr_i/dt = v_i

dv_i/dt = a_i(r) = sum_{j != i}  G * m_j * (r_j - r_i) / ( |r_j - r_i|^2 + eps^2 )^(3/2)
```

Energia potencial total:

```
U(r) = - sum_{i < j}  G * m_i * m_j / sqrt( |r_j - r_i|^2 + eps^2 )
```

Energia cinética total e energia total:

```
K(v) = sum_i  (1/2) * m_i * |v_i|^2
E    = K + U
```

Momento linear e momento angular totais:

```
P = sum_i m_i * v_i
L = sum_i m_i * (r_i x v_i)
```

### 1.2 Constantes físicas (herdadas de 2019, inalteradas)

| Símbolo | Identificador | Valor | Unidade |
|---|---|---|---|
| G | `G` | `6.67408e-11` | m³ kg⁻¹ s⁻² |
| m | `PARTICLE_MASS` | `1.0e9` | kg |

Todas as partículas têm massa idêntica `m`. A implementação deve, ainda assim, receber um vetor
`masses` de comprimento `N` (as fórmulas acima usam `m_j` explicitamente); a igualdade de massas é
uma propriedade da condição inicial, não do núcleo de força.

### 1.3 Consistência força–potencial (obrigatória)

A aceleração da Seção 1.1 é **exatamente** o gradiente do potencial da Seção 1.1:

```
a_i = - (1 / m_i) * dU/dr_i          [T]
```

Isto não é acidental: é o que torna o sistema regularizado um sistema hamiltoniano genuíno e,
portanto, o que torna `E` um invariante cuja conservação diz algo sobre o integrador e não sobre a
inconsistência do modelo. **A implementação deve preservar esta identidade.** Ela é testável
diretamente (invariante `INV-1`, Seção 7).

---

## 2. Semântica do softening

### 2.1 Qual potencial está sendo resolvido

A forma usada em 2019 — `dsq = dx² + dy² + dz² + sft²` com `G*m_j / (dsq * sqrt(dsq))` na
aceleração e `sqrt(... + sft²)` no potencial — **é exatamente o softening de Plummer**. Confirmado.

O núcleo de par `-G m_i m_j / sqrt(d² + eps²)` é, ponto a ponto, o potencial gerado por uma
distribuição de Plummer

```
rho_eps(x) = (3 m / (4 pi eps^3)) * (1 + |x|^2/eps^2)^(-5/2)
```

avaliado à distância `d` do seu centro. A interpretação física exata é: **cada partícula é tratada
como massa pontual movendo-se no campo de uma esfera de Plummer de escala `eps` centrada em cada
uma das outras**. O núcleo é simétrico em `(i, j)`, logo a terceira lei de Newton continua valendo
termo a termo, e `P` e `L` continuam sendo invariantes exatos do problema contínuo.

### 2.2 O que isso implica — dito com todas as letras

**O sistema simulado não é o problema newtoniano de massas pontuais, e não é kepleriano.** Para
`d >> eps` a diferença é `O(eps²/d²)`; para `d <~ eps` o sistema é qualitativamente distinto:

1. **A força de par é limitada.** O módulo da aceleração mútua de duas partículas atinge máximo em
   `d = eps/sqrt(2)`, com valor
   ```
   a_pair_max = (2/(3*sqrt(3))) * G*m / eps^2 = 0.3849 * G*m / eps^2
   ```
   Não há singularidade. Com `eps = 0.05 m` e `m = 1e9 kg`: `a_pair_max = 10.27 m/s²`.

2. **A energia potencial é limitada inferiormente:** `U >= - G m² N (N-1) / (2 eps)`. O sistema não
   pode colapsar a energia `-infinito`. Com os parâmetros da Seção 4: `U >= -6.67e14 J`.

3. **A frequência de par é limitada superiormente.** Perto de `d = 0` o potencial de par é
   harmônico, com
   ```
   omega_pair_max = sqrt( G*(m_i + m_j) / eps^3 )
   P_eps          = 2*pi / omega_pair_max
   ```
   Para qualquer par isolado, o período orbital é **maior ou igual** a `P_eps`: a constante elástica
   efetiva `Phi'(r)/r = mu/(r²+eps²)^(3/2)` decresce com `r`, logo o oscilador amolece com a
   amplitude e o período cresce monotonicamente, do limite harmônico ao limite kepleriano. Com
   `eps = 0.05 m`, `m = 1e9 kg`: `omega_pair_max = 32.678 rad/s`, `P_eps = 0.192276 s`. **[T]**

   Verificado por quadratura do período radial `T(r_max) = 4 ∫_0^{r_max} dr / sqrt(2(Phi(r_max)-Phi(r)))`
   **[M]**: `T/P_eps = 1.0002` para `r_max = 1e-3 m`, `1.022` para `0.01 m`, `1.503` para `0.05 m`,
   `2.70` para `0.1 m`, `63.7` para `1.0 m`. Monótono, sem mínimo interior.

   Este é o ponto decisivo do projeto: **é o softening que torna bem-posta a integração com passo
   global fixo.** Sem ele não existe cota inferior para a escala de tempo mais rápida do sistema, e
   nenhum `dt` fixo é justificável a priori. Qualquer discussão de ordem de convergência ou de
   conservação de energia com `eps -> 0` e `dt` fixo é vazia.

4. **Consequência para os testes analíticos:** a solução de Kepler **não** resolve o problema de dois
   corpos suavizado. Ver Seção 7, invariantes `INV-5` e `INV-6`: o teste kepleriano roda com
   `eps = 0` exatamente; o teste com softening usa a solução circular exata do potencial suavizado,
   que tem forma fechada própria.

### 2.3 Veredito sobre `sft = 1e-10` (valor de 2019)

`sft = 1e-10 m` **não é uma regularização física**. É um guarda contra divisão por zero no termo
`i = j`. Com esse valor:

- `a_pair_max = 0.3849 * 0.0667408 / 1e-20 = 2.57e18 m/s²`;
- `P_eps = 2*pi*sqrt(1e-30 / 0.1334816) = 5.44e-15 s`.

Ou seja, para integrar com passo fixo resolvendo o encontro mais próximo possível seriam necessários
`dt ~ 1e-16 s`. Isso era inofensivo em 2019 **apenas** porque a execução tinha `M = 1` passo a
partir de uma grade cúbica onde a separação mínima era exatamente `1.0 m` e nenhuma partícula chegava
perto de outra. **Com `sft = 1e-10`, qualquer execução longa a partir de uma condição inicial de
Poisson é lixo numérico**: o primeiro par que se aproximar recebe uma aceleração arbitrariamente
grande e é ejetado numericamente do sistema.

### 2.4 Decisão

**O projeto mantém a forma funcional de Plummer e altera apenas o valor de `eps`.**

Justificativa:

1. É consistente força–potencial (Seção 1.3) — condição necessária para que a conservação de energia
   seja um teste do integrador.
2. Preserva continuidade com o código de 2019: o núcleo `O(N²)` é literalmente o mesmo, sem
   ramificações.
3. Limita `omega_pair_max`, o que torna demonstrável a adequação de um `dt` fixo (Seção 4.4).
4. Alternativas de suporte compacto (spline de Monaghan–Lattanzio, por exemplo) são exatamente
   newtonianas além de `2h` — mais fiéis astrofisicamente —, mas introduzem ramificação no laço
   interno, o que penaliza a GPU por divergência de warp/wavefront e viola a restrição de não
   aumentar a carga computacional. Rejeitadas.

Valores de `eps` fixados na Seção 4. `eps` é um **parâmetro do modelo**, não um parâmetro de ajuste
numérico: ele deve constar de todo artefato de saída e de toda legenda de gráfico.

### 2.5 Requisito de implementação sobre o termo `i = j`

O núcleo deve permitir `eps = 0` (necessário para `INV-5`). Portanto a diagonal `i = j` deve ser
excluída **antes** de qualquer multiplicação por `d`, e nunca por meio de `eps` ser diferente de
zero.

Forma **preferida** (nenhum `inf` é produzido em momento algum):

```
dsq = |r_j - r_i|^2 + eps^2
dsq[i, i] = 1.0                 # valor arbitrário não nulo, apenas para a potência
inv = dsq ^ (-3/2)
inv[i, i] = 0.0
a_i = G * sum_j m_j * inv[i, j] * (r_j - r_i)
```

Forma **aceitável**: calcular `inv = dsq^(-3/2)` (a diagonal vira `inf` quando `eps = 0`), zerar a
diagonal de `inv`, e só então contrair com `d`.

Forma **inaceitável**: confiar em `dsq > 0` por causa de `eps`; ou multiplicar antes de zerar,
produzindo `inf * 0 = nan`; ou usar `eps` pequeno mas não nulo como substituto de excluir a
diagonal.

---

## 3. Os quatro integradores

Notação: passo `h = dt`; estado no passo `n` é o par síncrono `(r^n, v^n)` no instante `t_n = n*h`;
`A(r)` denota uma avaliação completa do campo de acelerações da Seção 1.1 (custo `O(N²)`).

Todos os quatro esquemas partem de `v^0 = 0` e de `r^0` dado pela Seção 5.

**Contabilidade de custo (normativa).** Cada método é caracterizado por **dois** números, não um:

```
n_force(method, M) = M * k_method + s_method
```

- `k_method` = avaliações de `A` **por passo** (regime estacionário do laço);
- `s_method` = avaliações de **pré-condição**, executadas uma única vez antes do laço, independentes
  de `M`.

Cada seção abaixo declara os dois. `s` é ignorável em execução longa e **não** é ignorável na
comparação a custo igual da Seção 8.2, que é onde `M` é pequeno de propósito. A separação em dois
números existe para que a contabilidade seja uniforme entre os quatro métodos, e não para que o
termo `s` possa ser esquecido.

Verificado por contagem instrumentada de chamadas a `accelerations`, `M ∈ {0, 1, 2, 8, 64}`, os
quatro métodos **[M]**: a fórmula acima é exata em todos os casos, inclusive `M = 0` (onde
`velocity_verlet` custa `1` avaliação e os outros três custam `0`).

### 3.1 Euler explícito — `euler`

```
a^n     = A(r^n)
r^(n+1) = r^n + h * v^n
v^(n+1) = v^n + h * a^n
```

- Avaliações de força: `k = 1` por passo, `s = 0` de pré-condição. Custo total de `M` passos:
  **exatamente `M`**. Não há reaproveitamento possível: `a^n` é consumido dentro do próprio passo e
  `a^(n+1) = A(r^(n+1))` exige o estado novo. **[T]**
- Ordem de convergência global: **1**. **[T]**
- Simplético: **não**. O jacobiano do mapa tem determinante `1 + h² * (autovalores do tensor de
  maré)`; a forma simplética não é preservada.
- Reversível no tempo: não.
- Conserva `L` exatamente: **não**. Algebricamente,
  `sum_i r_i^(n+1) x v_i^(n+1) = sum_i r_i^n x v_i^n + h² * sum_i v_i^n x a_i^n`, e o termo `O(h²)`
  não se anula. **[T]**
- Conserva `P` a menos de arredondamento: **sim** (`sum_i m_i v_i^(n+1) = sum_i m_i v_i^n + h * sum_i m_i a_i^n`
  e `sum_i m_i a_i = 0` por antissimetria de pares). **[T]**
- Comportamento de energia em execução longa: **crescimento secular monótono e ilimitado**. Para o
  oscilador harmônico o fator de amplificação por passo é `sqrt(1 + omega²h²) > 1` para todo `h > 0`
  — o método é **incondicionalmente instável** em problemas oscilatórios; refinar `h` adia, não
  evita. **[T]**

  Medido (par de dois corpos ligado, `eps = 0.05`, `a = 1 m`, `e = 0.5`, 20 períodos): **[M]**

  | passos/período | `ΔE/|E₀|` final | `ΔL/|L₀|` |
  |---|---|---|
  | 1000 | `+6.97e-01` | `3.4e-01` |
  | 2000 | `+5.34e-01` | `2.4e-01` |
  | 4000 | `+4.00e-01` | `1.7e-01` |
  | 8000 | `+2.75e-01` | `1.1e-01` |

  Note que oito vezes mais passos reduzem o erro de energia por um fator inferior a 3. Esta é a
  assinatura que o estudo deve exibir.

**Armadilha de implementação (normativa):** `r^(n+1)` usa `v^n`. Escrever `v` antes de `r` sem
variável temporária transforma o esquema em Euler semi-implícito silenciosamente. Usar temporário.

### 3.2 Euler semi-implícito (simplético / Euler–Cromer) — `symplectic_euler`

```
a^n     = A(r^n)
v^(n+1) = v^n + h * a^n
r^(n+1) = r^n + h * v^(n+1)
```

Esta é a variante **"velocidade primeiro"**. É o membro exato da família implementado, e é o adjunto
da variante "posição primeiro".

- Avaliações de força: `k = 1` por passo, `s = 0` de pré-condição. Custo total de `M` passos:
  **exatamente `M`**, pelo mesmo argumento de 3.1. **[T]**
- Ordem de convergência global: **1** em posição. **[T]** Em **velocidade**, o expoente medido nesta
  configuração é **2**, não 1 — ver `INV-8`, ressalva 2. **[M]**
- Simplético: **sim**, exatamente (o mapa é composição de dois cisalhamentos, cada um de jacobiano
  unitário e triangular). **[T]**
- Reversível no tempo: **não** (o adjunto é a outra variante).
- Conserva `L` **exatamente** (a menos de arredondamento):
  `r^(n+1) x v^(n+1) = (r^n + h v^(n+1)) x v^(n+1) = r^n x v^(n+1) = r^n x v^n + h * (r^n x a^n)`,
  e `sum_i r_i x a_i = 0` porque as forças são centrais e obedecem à terceira lei. **[T]**
- Conserva `P` a menos de arredondamento: **sim**. **[T]**
- Comportamento de energia em execução longa: **oscilação limitada, sem deriva secular**. O método
  conserva exatamente um hamiltoniano sombra `H_h = H + O(h)`; portanto `E` oscila com amplitude
  `O(h)` em torno de um valor deslocado, e não deriva. **[T]** — válido enquanto `h` for pequeno
  frente a todas as escalas de tempo do sistema (Seção 4.4); a garantia é assintótica, não
  incondicional.

  Medido (mesmo par ligado, 20 períodos): **[M]**

  | passos/período | amplitude `max ΔE/|E₀| - min ΔE/|E₀|` | razão | `ΔE/|E₀|` final | `ΔL/|L₀|` |
  |---|---|---|---|---|
  | 1000 | `1.692e-02` | — | `+8.8e-05` | `8.7e-15` |
  | 2000 | `8.461e-03` | 2.00 | `+4.4e-05` | `1.5e-14` |
  | 4000 | `4.231e-03` | 2.00 | `+2.2e-05` | `7.0e-15` |
  | 8000 | `2.115e-03` | 2.00 | `+1.1e-05` | `5.7e-16` |

  A amplitude escala como `h¹` (razão 2.00 por refinamento de 2), e o valor final é ~200 vezes menor
  que a amplitude — isto é oscilação, não deriva.

  **Atenção:** numa órbita **circular** o termo `O(h)` do hamiltoniano sombra se anula por simetria
  (`v . a = 0` ao longo da órbita) e a amplitude medida escala como `h²`, imitando Verlet. Por isso
  o teste de ordem da amplitude de energia usa órbita **excêntrica** (`INV-7`).

**Este é o esquema de 2019.** Confirmado por inspeção dos quatro núcleos do notebook
`legacy/notebooks-2019/NBody_M1_N32Kcube_Python_x_NumbaCPU_x_NumbaGPU64_Neuromancer_v1.6.4.ipynb`:

- `gravitationalNbody_NumbaCPU` (escalar): o laço `i` acumula `a[i]` a partir de `r` e atualiza
  `v[i]`; um **segundo** laço `i` atualiza `r[i]`. Como `r` só é escrito no segundo laço, todas as
  acelerações usam `r^n`. Resultado: `v^(n+1) = v^n + h a(r^n)`, `r^(n+1) = r^n + h v^(n+1)`.
- `gravitationalNbody_NumPy_NumbaCPU` (vetorizado): `a = zeros`; laço `i` acumula
  `a[j] += G*m[i]*(r[i]-r[j])/(...)`, que é a aceleração sobre `j` devida a `i` — correto; depois
  `v += a*dt` e `r += v*dt`. Mesmo esquema.
- Núcleos CUDA (`..._a_v_ij_...` seguido de `..._r_i_...` com `cuda.synchronize()` entre eles):
  mesma separação em duas fases. Mesmo esquema.

**Veredito: sim, o esquema de 2019 é Euler semi-implícito (simplético), variante velocidade
primeiro.** Não é Euler explícito. Não é Verlet.

**Armadilha de implementação (normativa):** a separação em dois laços é o que garante o esquema. Se
alguém "otimizar" fundindo a atualização de `r` no primeiro laço, o resultado passa a ser uma
iteração tipo Gauss–Seidel — cada partícula `i` sente as partículas `j < i` já deslocadas —, que não
é nenhum dos quatro métodos, não é simplética e cuja ordem depende da ordem de indexação. A forma
vetorizada é imune a isso por construção; a forma escalar não é.

### 3.3 Verlet de velocidade — `velocity_verlet` (forma KDK, com reaproveitamento)

Forma normativa. **Kick–Drift–Kick**, com `a` mantido entre passos:

```
pré-condição: a^0 = A(r^0), calculado uma única vez antes do laço

v^(n+1/2) = v^n       + (h/2) * a^n
r^(n+1)   = r^n       + h * v^(n+1/2)
a^(n+1)   = A(r^(n+1))                     <-- única avaliação de força do passo
v^(n+1)   = v^(n+1/2) + (h/2) * a^(n+1)
```

- Avaliações de força: `k = 1` por passo, **`s = 1` de pré-condição**. Custo total de `M` passos:
  **`M + 1`**, não `M`. **[T]** Ver a nota dedicada abaixo — este é o único dos quatro métodos com
  `s != 0`, e a contagem correta é normativa para a Seção 8. O reaproveitamento é **obrigatório**:
  sem ele o custo seria `2M`, e aí sim a comparação da Seção 8 ficaria errada por um fator, não por
  um termo de borda.
- Ordem de convergência global: **2**. **[T]**
- Simplético: **sim**, exatamente (composição kick–drift–kick, cada fator com jacobiano unitário).
  **[T]**
- Reversível no tempo: **sim** (o mapa é simétrico). **[T]**
- Conserva `L` **exatamente** (a menos de arredondamento): cada um dos três subpassos preserva
  `sum_i r_i x v_i`, pelo mesmo argumento de `sum_i r_i x a_i = 0`. **[T]**
- Conserva `P` a menos de arredondamento: **sim**. **[T]**
- Comportamento de energia em execução longa: **oscilação limitada de amplitude `O(h²)`, sem deriva
  secular**, sob a mesma ressalva assintótica do item anterior. **[T]**

  Medido (mesmo par ligado, 20 períodos): **[M]**

  | passos/período | amplitude | razão | `ΔE/|E₀|` final | `ΔL/|L₀|` |
  |---|---|---|---|---|
  | 1000 | `1.012e-04` | — | `+2.9e-09` | `5.3e-15` |
  | 2000 | `2.530e-05` | 4.00 | `+7.2e-10` | `1.4e-14` |
  | 4000 | `6.324e-06` | 4.00 | `+1.8e-10` | `7.5e-15` |
  | 8000 | `1.581e-06` | 4.00 | `+4.5e-11` | `2.9e-14` |

**Equivalência algébrica, não binária.** A forma "posição"
`r^(n+1) = r^n + h v^n + (h²/2) a^n ; v^(n+1) = v^n + (h/2)(a^n + a^(n+1))` é o mesmo método em
aritmética exata, mas **não** produz os mesmos bits em ponto flutuante. Nenhum teste pode exigir
igualdade binária entre as duas formas. Se ambas forem implementadas, a tolerância entre elas é a de
`TOL-IMPL` (Seção 6.3).

**Nota normativa — o custo é `M + 1`, e isso não é um defeito.**

A cadeia de reaproveitamento `a^0 -> a^1 -> ... -> a^M` precisa ser semeada: `a^0 = A(r^0)` é uma
avaliação genuína de `A`, com o mesmo custo `O(N²)` de qualquer outra, e sem ela o método **não está
definido**. `M` passos custam `M + 1` avaliações. Para `RUN_COLLAPSE`: `12601`, não `12600`.

Três esclarecimentos, todos necessários porque o ponto é sutil e já foi confundido:

1. **Nenhuma avaliação é desperdiçada.** As `M + 1` são todas consumidas por uma atualização de
   velocidade: `a^n` entra no *kick* final do passo `n` e no *kick* inicial do passo `n+1`. O termo
   `+1` é o preço de o esquema ser síncrono nas pontas, não uma ineficiência da implementação.
2. **O termo `+1` é propriedade do método, não da forma algébrica.** A forma "posição" acima também
   exige `a^0` antes do laço e também custa `M + 1`. Reescrever o esquema não elimina a semeadura.
3. **O custo não afeta a trajetória.** A trajetória produzida por `M` passos é função apenas de
   `(r^0, v^0, h, M)`. A avaliação de pré-condição não altera `h`, não altera `M`, não altera
   `err_r` nem `err_v`, e portanto **não altera nenhum expoente de convergência de `INV-8`** — os
   expoentes são medidos contra `dt`, não contra `n_force`. Em particular, o valor pré-assintótico
   `p_r = 4.297` na razão `64 -> 128` (`INV-8`, ressalva 1) é inteiramente independente desta
   contabilidade. O que muda é exclusivamente a **abscissa** dos pontos de Verlet no gráfico da
   Seção 8.3. Confundir as duas coisas é o erro que esta nota existe para impedir.

### 3.4 Runge–Kutta de 4ª ordem — `rk4`

Aplicado ao sistema de primeira ordem `y = (r, v)`, `dy/dt = f(y) = (v, A(r))`. Forma normativa
(RK4 clássico, tableau de Kutta):

```
k1_r = v^n                      ;  k1_v = A( r^n )
k2_r = v^n + (h/2) * k1_v       ;  k2_v = A( r^n + (h/2) * k1_r )
k3_r = v^n + (h/2) * k2_v       ;  k3_v = A( r^n + (h/2) * k2_r )
k4_r = v^n +  h    * k3_v       ;  k4_v = A( r^n +  h    * k3_r )

r^(n+1) = r^n + (h/6) * ( k1_r + 2*k2_r + 2*k3_r + k4_r )
v^(n+1) = v^n + (h/6) * ( k1_v + 2*k2_v + 2*k3_v + k4_v )
```

**Atenção ao acoplamento cruzado:** o argumento de `A` no estágio `k(n+1)_v` é construído com
`k(n)_r` (incremento de posição), não com `k(n)_v`. Trocar isso produz um método consistente de
ordem 2 que passa despercebido em testes frouxos.

- Avaliações de força: `k = 4` por passo, `s = 0` de pré-condição. Custo total de `M` passos:
  **exatamente `4M`**. **[T]** Vale registrar por que não há termo de borda, já que RK4 é o candidato
  natural a tê-lo: o tableau clássico **não é FSAL**. O vetor de pesos é
  `b = (1/6, 1/3, 1/3, 1/6)` e a última linha de `A` é `(0, 0, 1, 0)`; como `b != A[4,:]`, o
  argumento do quarto estágio `r^n + h*k3_r` **não** é `r^(n+1)`, e nenhum estágio pode ser
  transportado para o passo seguinte. Não há o que semear nem o que reaproveitar.
- Ordem de convergência global: **4**. **[T]**
- Simplético: **não**. Nenhum RK explícito de 4 estágios é simplético.
- Reversível no tempo: não.
- Conserva `L` exatamente: **não**; o erro em `L` decai como `O(h⁴)` ou mais rápido. **[M]**
- Conserva `P` a menos de arredondamento: **sim** (cada estágio tem `sum_i m_i k_v = 0`). **[T]**
- Comportamento de energia em execução longa: **deriva secular**, de magnitude muito pequena mas
  monotônica e proporcional ao tempo decorrido. O sinal observado nesta configuração é **negativo**
  (dissipação aparente). **[M]**

  Medido (mesmo par ligado, 20 períodos): **[M]**

  | passos/período | amplitude | razão | `ΔE/|E₀|` final | `ΔL/|L₀|` |
  |---|---|---|---|---|
  | 1000 | `6.739e-09` | — | `-5.24e-09` | `8.2e-10` |
  | 2000 | `2.615e-10` | 25.8 | `-1.64e-10` | `2.6e-11` |
  | 4000 | `1.134e-11` | 23.1 | `-5.12e-12` | `8.1e-13` |
  | 8000 | `6.493e-13` | 17.5 | `-2.58e-13` | `7.6e-14` |

  A razão decresce de ~26 para ~17 conforme o piso de arredondamento fp64 é atingido; o teste deve
  exigir razão `>= 12` (ordem efetiva `>= 3.6`), não um valor exato.

  Diferença qualitativa em relação aos simpléticos: em Verlet e Euler simplético o valor **final** é
  ordens de grandeza menor que a amplitude da oscilação; em RK4 o valor final é **igual em magnitude**
  ao extremo da excursão, porque a excursão é a deriva.

### 3.5 Tabela resumo

| método | `k` (por passo) | `s` (pré-cond.) | custo de `M` passos | ordem nominal | ordem medida (`r` / `v`) | simplético | reversível | `L` exato | `P` exato* | energia em execução longa |
|---|---|---|---|---|---|---|---|---|---|---|
| `euler` | 1 | 0 | `M` | 1 | `→1⁻` / `→1⁻` | não | não | não | sim | crescimento secular ilimitado |
| `symplectic_euler` | 1 | 0 | `M` | 1 | `1.00` / **`2.00`** | **sim** | não | **sim** | sim | oscilação limitada, amplitude `O(h)` |
| `velocity_verlet` | 1 | **1** | **`M + 1`** | 2 | `2.00` / `2.00` | **sim** | **sim** | **sim** | sim | oscilação limitada, amplitude `O(h²)` |
| `rk4` | 4 | 0 | `4M` | 4 | **`4.95`** / **`4.95`** | não | não | não | sim | deriva secular, negativa |

A coluna "ordem medida" é o valor **[M]** desta configuração (`INV-8`) e é a que vale para os
testes. As duas células em negrito divergem da ordem nominal e são discutidas nas ressalvas 2 e 4 de
`INV-8`; ignorá-las produz testes que falham contra implementações corretas.

As colunas `k` e `s` são **ambas** normativas e devem ser expostas pelo contrato de API como duas
tabelas separadas — `FORCE_EVALS_PER_STEP` (a coluna `k`) e `FORCE_EVALS_STARTUP` (a coluna `s`).
`FORCE_EVALS_PER_STEP` sozinho não é suficiente para calcular o custo de uma execução e não deve ser
usado como se fosse: quem multiplicar `n_steps * FORCE_EVALS_PER_STEP[m]` e chamar o resultado de
custo estará subestimando `velocity_verlet`. Valores em números na Seção 9.

\* "exato" aqui significa exato em aritmética exata e ao nível do arredondamento em ponto flutuante;
ver Seção 7, `INV-2`.

### 3.6 Limites de estabilidade

Para o modo mais rápido do sistema, de frequência `omega_max`:

| método | condição de estabilidade linear | `h_max` com `omega_max = 42 rad/s` (Seção 4.4) |
|---|---|---|
| `euler` | nenhuma (instável para todo `h > 0`) | — |
| `symplectic_euler` | `omega * h < 2` | `0.0476 s` |
| `velocity_verlet` | `omega * h < 2` | `0.0476 s` |
| `rk4` | `omega * h < 2*sqrt(2) = 2.828` | `0.0673 s` |

Estabilidade não é precisão. Os `dt` da Seção 4 são duas ordens de grandeza menores que esses
limites; isso é deliberado.

---

## 4. Regime dinâmico e parâmetros fixados

### 4.1 Por que esfera fria

A grade cúbica homogênea em repouso de 2019 não é um problema de validação: não tem solução
analítica, não tem escala de tempo característica com forma fechada, e com `M = 1` passo nunca saiu
do estado inicial. A esfera fria de densidade uniforme em repouso é o problema-teste clássico
(*cold collapse*), com tempo de queda livre analítico

```
t_ff = sqrt( 3*pi / (32 * G * rho) )
```

que fornece um alvo quantitativo contra o qual medir a simulação.

**Custo computacional: inalterado.** A amostragem da esfera é `O(N)`, executada uma vez. O núcleo de
força permanece `O(N²)` idêntico, sem ramificações adicionais. Para o mesmo `N`, o custo por passo é
bit a bit o mesmo trabalho aritmético do código de 2019.

### 4.2 Escalas físicas com os valores de 2019

Preservando `m = 1e9 kg`, `N = 1000` e a densidade numérica da grade de 2019 (espaçamento `1.0 m`,
isto é, `n = 1 partícula/m³`), o raio da esfera equivalente é

```
R_0 = ( 3*N / (4*pi) )^(1/3) = 6.2035049090 m
```

o que dá exatamente o mesmo volume (`1000 m³`), a mesma massa total (`1e12 kg`) e a mesma densidade
(`1e9 kg/m³`) da grade cúbica de 2019. **Esta é a razão da escolha de `R_0`:** a esfera é a
reorganização geométrica da mesma matéria, sem mudança de regime.

| grandeza | expressão | valor |
|---|---|---|
| massa total `M_tot` | `N*m` | `1.0e12 kg` |
| densidade `rho` | `M_tot / (4/3 pi R_0³)` | `1.0e9 kg/m³` |
| tempo de queda livre `t_ff` | `sqrt(3 pi/(32 G rho))` | **`2.1007035 s`** |
| tempo dinâmico `t_dyn` | `sqrt(R_0³/(G M_tot))` | `1.8912976 s` |
| velocidade característica | `sqrt(G M_tot/R_0)` | `3.2800 m/s` |
| aceleração na borda | `G M_tot / R_0²` | `1.7343 m/s²` |
| separação interpartícula média | `n^(-1/3)` | `1.0 m` |
| aceleração de par a `1 m` | `G m / 1²` | `0.0667408 m/s²` |
| razão campo médio / par | — | `26.0` |
| `U` contínuo (esfera uniforme) | `-3 G M_tot²/(5 R_0)` | `-6.45514e12 J` |

**A estimativa preliminar do enunciado está correta:** `rho ~ 1e9 kg/m³`, `t_ff = 2.10 s`, e com
`dt = 0.01` isso são 210 passos. Ver, porém, a Seção 4.4: `dt = 0.01` **não** é um passo admissível
para atravessar o colapso, por razões independentes da contagem de passos até `t_ff`.

### 4.3 Escolha de `eps`

Critérios, todos verificados numericamente nesta configuração:

1. **Não perturbar a estrutura inicial.** `eps/R_0 = 8.06e-3`. A energia potencial inicial difere da
   não-suavizada por `3.4e-4` em termos relativos. **[M]**
2. **Resolver o núcleo colapsado.** No instante de máxima compressão o raio de meia-massa é
   `r_half,min = 0.3472 m` **[M]**; a densidade numérica local no núcleo é `~1.4e3 m⁻³`, isto é,
   separação interpartícula local `~0.089 m`. Com `eps = 0.05 m`, o softening vale `~0.56` da
   separação local no instante mais denso. **Este é o limite de resolução do estudo e deve ser
   declarado como tal:** a estrutura do núcleo abaixo de `~0.05 m` não é resolvida.
3. **Limitar a frequência máxima** (Seção 2.2, item 3), viabilizando `dt` fixo.
4. **Robustez do resultado ao valor de `eps`.** Verificado **[M]** varrendo `eps ∈ {0.02, 0.05, 0.1}`
   (fator 5) e `dt ∈ {2e-3, 1e-3, 5e-4, 2.5e-4}` (fator 8), 12 execuções de Verlet fp64 no colapso
   completo:

   | `eps` | `t_collapse / t_ff` (parábola) | `r_half,min` |
   |---|---|---|
   | `0.02` | `1.0346 – 1.0360` | `0.3438 – 0.3486 m` |
   | `0.05` | `1.0361` (todos os `dt`) | `0.3472 m` (todos os `dt`) |
   | `0.10` | `1.0371` (todos os `dt`) | `0.3600 m` (todos os `dt`) |

   O instante de colapso varia `0.25%` sobre um fator 5 em `eps`; `r_half,min` varia `4.5%`. Os
   observáveis de validação **não** são artefatos do softening.

**Valor fixado: `eps = 0.05 m` (`SOFTENING = 5.0e-2`).**

Valor alternativo permitido para estudo de sensibilidade, **não** para os testes: `eps = 0.02 m`.
Pelo mesmo critério da Seção 4.4 ele exige `dt <= 2.0e-4 s` (`P_eps` cai para `0.0486 s`), isto é,
`M >= 31500` passos — 2,5 vezes o custo de `RUN_COLLAPSE`.

### 4.4 Escolha de `dt` — critério de resolução

A frequência máxima presente no sistema é limitada superiormente por

```
omega_max <= sqrt( G * (1 + K) * m / eps^3 )
```

onde `K` é o número máximo de vizinhos dentro de `eps` de uma mesma partícula. Medido ao longo de
todo o colapso **[M]**:

| fase | `K` máximo | `omega_max` medido (maior autovalor do tensor de maré) |
|---|---|---|
| antes do colapso (`t < 0.8 t_ff`) | 1–2 | `22–26 rad/s` |
| durante e após o colapso | 3–5 | `33–42 rad/s` |

A cota de par isolado (`K = 1`) dá `32.68 rad/s`; a cota com `K = 4` dá `51.7 rad/s`. O máximo
efetivamente observado é **`41.9 rad/s`**. Adota-se `omega_max = 42 rad/s` como valor de projeto.

Critério adotado: **pelo menos 250 passos por período do modo mais rápido**, isto é

```
dt <= (2*pi) / (250 * omega_max) = 5.98e-4 s
```

**Valor fixado: `dt = 5.0e-4 s`** (`299 passos/período` no pior instante medido; margem de 95× sobre
o limite de estabilidade de Verlet).

Verificação do critério — pico de `|ΔE/E₀|` de Verlet fp64 ao longo do colapso completo,
`eps = 0.05`: **[M]**

| `dt` | pico `|ΔE/E₀|` | razão | `ΔE/E₀` final | `r_half,min` | `t_collapse/t_ff` |
|---|---|---|---|---|---|
| `2.0e-3` | `3.59e-3` | — | `+1.31e-4` | `0.3472` | `1.0361` |
| `1.0e-3` | `9.17e-4` | 3.91 | `-1.66e-5` | `0.3472` | `1.0361` |
| `5.0e-4` | `2.29e-4` | 4.00 | `-3.80e-6` | `0.3472` | `1.0361` |

O pico escala como `h²` (assinatura de Verlet). Os observáveis do colapso já estão convergidos em
`dt = 2e-3`; `dt = 5e-4` é escolhido pelo critério de energia, não pelo de trajetória.

**Por que `dt = 0.01` (valor de 2019) é inadmissível:** `omega_max * dt = 0.42`, isto é, 15 passos
por período do modo mais rápido. O erro de energia de Verlet seria `~(0.42)²/4 ≈ 4%` por modo rápido,
e o colapso seria atravessado às cegas. `dt = 0.01` permanece válido apenas para os *benchmarks de
tempo* com `M = 1` (Seção 4.7), que não fazem afirmação física alguma.

### 4.5 Conjunto de parâmetros `RUN_COLLAPSE` — colapso completo

Uso: comportamento de energia em execução longa, medição de `t_collapse`, comparação qualitativa
entre integradores, figuras do colapso.

| parâmetro | identificador | valor |
|---|---|---|
| número de corpos | `N` | `1000` |
| massa por corpo | `PARTICLE_MASS` | `1.0e9 kg` |
| raio inicial | `SPHERE_RADIUS` | `6.2035049090 m` |
| softening | `SOFTENING` | `5.0e-2 m` |
| passo | `DT` | `5.0e-4 s` |
| número de passos | `N_STEPS` | `12600` |
| tempo total | `T_END` | `6.30 s = 2.99899 t_ff` |
| semente | `SEED` | `20190222` |
| cadência de saída | `OUT_DT` | `1.0e-2 s` (a cada 20 passos) |

`T_END = 3 t_ff` cobre a queda livre, o *bounce* e a virialização parcial. Custo, pela fórmula da
Seção 8.1: `12600` avaliações de força para `euler` e `symplectic_euler`, **`12601`** para
`velocity_verlet` (a avaliação de pré-condição, Seção 3.3) e `50400` para `rk4` (mesmo `dt`). O termo
de borda vale `0.008%` aqui e é registrado por uniformidade de contabilidade, não por relevância.

### 4.6 Conjunto de parâmetros `RUN_CONVERGENCE` — ordem de convergência

Uso: medição da ordem empírica e da curva erro × custo. Intervalo escolhido **antes** do colapso,
onde a solução é suave e não houve encontro próximo — condição necessária para que o erro esteja no
regime assintótico.

| parâmetro | valor |
|---|---|
| condição inicial | idêntica a `RUN_COLLAPSE` (mesma semente) |
| `SOFTENING` | `5.0e-2 m` |
| `T_END` | `T_conv = 0.5 * t_ff = 1.05035175 s` |
| escada de refinamento | `n_steps ∈ {64, 128, 256, 512, 1024, 2048}` para `symplectic_euler`, `velocity_verlet` e `rk4`; estendida a `{…, 4096}` para `euler` |
| razão de refinamento | `2` |
| níveis | `6` (5 razões de erro); `7` para `euler` |
| primeira razão utilizável | `n = 256` (`velocity_verlet`, `symplectic_euler`), `n = 128` (`rk4`) — ver `INV-8`, ressalva 1 |
| referência | ver Seção 5 |

Justificativa do intervalo: em `t = 0.5 t_ff` o raio de meia-massa caiu apenas de `4.881` para
`~4.14 m` e a razão virial `-2K/U` vale `~0.33` **[M]** — regime de queda ainda essencialmente
laminar.

### 4.7 Conjunto de parâmetros `RUN_BENCH` — benchmark de desempenho

**Inalterado em relação a 2019**, e explicitamente fora do escopo físico:
`M = 1` passo, `dt = 0.01 s`, `N` variável (potências de dois e/ou `n_cube_edge³`). Nenhuma
afirmação sobre física, energia ou trajetória pode ser feita a partir destas execuções. Elas medem
throughput do núcleo `O(N²)` e nada mais.

**Nota explícita sobre carga computacional:** a restrição "não aumentar a carga computacional"
é satisfeita no sentido em que foi dada — mesmo `N`, mesmo `O(N²)`, mesma aritmética por par.
Ela **não** pode ser lida como "manter `M = 1`": não existe estudo de integradores com um único
passo. `RUN_COLLAPSE` executa `12600` passos. Com `N = 1000` isso são `1.26e10` interações de par
por integrador de 1 avaliação — segundos em GPU, minutos em CPU vetorizada. O custo total do estudo
físico é irrelevante frente ao benchmark de `N = 32768`, que faz `1.07e9` interações **por passo**.

### 4.8 Conjuntos de dois corpos

| identificador | `eps` | configuração | período |
|---|---|---|---|
| `TWOBODY_KEPLER` | `0.0` | `a = 1.0 m`, `e = 0.5`, início no apoastro | `17.19765239 s` |
| `TWOBODY_CIRC` | `0.05` | circular, separação `d = 1.0 m` | `17.22988792 s` |
| `TWOBODY_ECC` | `0.05` | `a = 1.0 m`, `e = 0.5`, início no apoastro | `17.19765239 s` (nominal) |

Detalhes na Seção 7, `INV-5`, `INV-6` e `INV-7`.

---

## 5. Condição inicial: amostragem da esfera fria

### 5.1 Distribuição alvo

Densidade de massa uniforme em `|x| <= R_0`, todas as partículas em repouso (`v_i = 0`), massas
iguais. Para densidade uniforme, a densidade de probabilidade radial é

```
p(r) dr proporcional a r² dr   =>   CDF(r) = (r/R_0)³
```

**O erro clássico é amostrar `r` uniformemente em `[0, R_0]`**, o que produz uma esfera com
`rho(r) ∝ 1/r²` — cúspide central, densidade não uniforme, `t_ff` errado e colapso qualitativamente
diferente. A inversa correta da CDF é `r = R_0 * u^(1/3)`.

### 5.2 Algoritmo normativo

Determinístico e reprodutível, exatamente nesta ordem de sorteios:

```
rng = numpy.random.default_rng(SEED)          # PCG64

u        = rng.random(N)                      # 1o bloco de N sorteios
radius   = SPHERE_RADIUS * numpy.cbrt(u)

cos_th   = 2.0 * rng.random(N) - 1.0          # 2o bloco de N sorteios
phi      = 2.0 * numpy.pi * rng.random(N)     # 3o bloco de N sorteios
sin_th   = numpy.sqrt(1.0 - cos_th**2)

positions[:, 0] = radius * sin_th * numpy.cos(phi)
positions[:, 1] = radius * sin_th * numpy.sin(phi)
positions[:, 2] = radius * cos_th

positions -= positions.mean(axis=0)           # recentragem no centro de massa
velocities = zeros((N, 3))
```

Pontos normativos:

- **Direção isotrópica por `cos(theta)` uniforme em `[-1, 1]`**, não `theta` uniforme em `[0, pi]`
  (este último concentra pontos nos polos).
- **Sem rejeição por separação mínima.** Verificado **[M]**: impor `d_min = 0.15 m` por
  rejeição altera `r_half,min` de `0.3472` para `0.3483` (`0.3%`) e não muda o comportamento de
  energia. A rejeição introduz correlação de par artificial sem benefício; fica proibida.
- **A recentragem é feita nas posições, não nas velocidades** — as velocidades são exatamente zero,
  logo `P = 0` exatamente na representação de ponto flutuante.
- Após a recentragem, `|r_i|` pode exceder ligeiramente `R_0` (o centro de massa amostral não
  coincide com o centro geométrico). Isso é esperado e correto; não "corrigir".
- A amostragem é feita **sempre em fp64**, mesmo quando a integração roda em fp32. A conversão para
  fp32 é feita depois, uma única vez. Isto garante que fp32 e fp64 partam da mesma condição inicial
  a menos de um arredondamento.

### 5.3 Semente e reprodutibilidade

`SEED = 20190222` (data da versão 1.6.5 do notebook de 2019). Semente fixa e obrigatória para todos
os resultados publicados.

A condição inicial gerada deve ser **gravada uma vez** em
`data/ic_sphere_N1000_seed20190222.npz` (arrays `positions`, `velocities`, `masses`, em fp64),
acompanhada do seu SHA-256 registrado no repositório. Motivo: a política de compatibilidade de fluxo
do `numpy.random.Generator` admite mudanças entre versões maiores. Os testes de física carregam o
arquivo; um teste separado e explicitamente marcado verifica que o amostrador reproduz o arquivo bit
a bit e falha ruidosamente se o fluxo do NumPy mudar.

### 5.4 Valores de referência da condição inicial (`SEED = 20190222`, `N = 1000`, fp64) **[M]**

| grandeza | valor | tolerância de teste |
|---|---|---|
| `max_i \|r_i\|` | `6.323302 m` | `1e-6` relativo |
| `r_half(0)` (raio de meia-massa) | `4.881251 m` | `1e-6` relativo |
| `r_half(0)` analítico `R_0 / 2^(1/3)` | `4.923700 m` | flutuação amostral de `0.86%` — **não** é discrepância |
| separação mínima de par | `0.0574 m` | `1e-4` relativo |
| separação ao vizinho mais próximo, mediana | `0.5695 m` | `1e-3` relativo |
| `\|sum_i r_i\|` após recentragem | ver `TOL-CENTER` abaixo | adimensional |
| `U(r⁰)` com `eps = 0.05` | `-6.4260397026e12 J` | ver `TOL-ENERGY` |
| `U(r⁰)` com `eps = 0.02` | `-6.4278154023e12 J` | ver `TOL-ENERGY` |
| `max_i \|a_i(r⁰)\|`, `eps = 0.05` | `9.075891 m/s²` | `1e-10` relativo (fp64) |
| `rms_i \|a_i(r⁰)\|`, `eps = 0.05` | `1.544392 m/s²` | `1e-10` relativo (fp64) |
| `K(v⁰)` | `0.0` exatamente | binário |
| `P(t=0)` | `0.0` exatamente | binário |

### 5.5 Resíduo da recentragem — `TOL-CENTER`

**Correção normativa.** Uma versão anterior desta seção fixava `|sum_i r_i| < 1e-13 m`, um valor
**absoluto em metros**. Isso violava a nota normativa da Seção 6.3 ("nenhum teste deve comparar
valores absolutos em joules ou metros sem normalizar") e era, além disso, inatingível: `1e-13 m`
equivale a `72.6 * eps_fp64 * R_0`, uma exigência de `~72` ULPs sobre uma redução de `N = 1000`
parcelas. A cota é substituída pela forma adimensional abaixo. Nenhum resultado publicável foi
produzido sob a cota antiga.

**Enunciado.** O algoritmo da Seção 5.2 subtrai a média amostral, logo `sum_i r_i = 0` em aritmética
exata. **[T]** Em ponto flutuante não é exato, por duas razões independentes:

1. a média `positions.mean(axis=0)` é ela própria arredondada, e o erro `delta` na média reaparece
   multiplicado por `N` no somatório recentrado;
2. cada subtração `r_i - mean` é arredondada individualmente.

Ambas as contribuições escalam como `eps_prec * sum_i |r_i|`, isto é, **linearmente em `N` e
linearmente numa escala de comprimento**. Essa é a forma da cota. Adota-se, com a mesma estrutura de
`TOL-MOM` (Seção 6.3, `INV-2`):

```
TOL-CENTER :   || sum_i r_i ||  /  ( N * eps_prec * R_0 )   <=  1
```

com `R_0 = SPHERE_RADIUS = 6.2035049090 m` e `eps_prec ∈ {eps_fp64 = 2.220446e-16,
eps_fp32 = 1.1920929e-07}`. O denominador é uma **constante da configuração**, não uma grandeza
medida: se o amostrador estiver quebrado, `max_i |r_i|` se move junto com o numerador e a razão
perde poder discriminante; `R_0` não se move. Denominador em fp64: `1.377455e-12 m`.

**Valores medidos** (`SEED = 20190222`, `N = 1000`, NumPy 2.5.1, `torch 2.9.1`) **[M]**:

| precisão | forma de somar | `\|\| sum_i r_i \|\|` | razão `TOL-CENTER` |
|---|---|---|---|
| fp64 | `numpy.sum` (pairwise) | `1.6037e-13 m` | `0.116` |
| fp64 | acumulação sequencial | `1.6968e-13 m` | `0.123` |
| fp64 | ordem invertida | `1.9453e-13 m` | `0.141` |
| fp64 | ordem embaralhada | `1.6540e-13 m` | `0.120` |
| fp64 | 7 blocos parciais | `1.7955e-13 m` | `0.130` |
| fp64 | `torch.Tensor.sum` | `1.9211e-13 m` | `0.140` |
| fp64 | `math.fsum` (soma exata do array armazenado) | `1.6572e-13 m` | `0.120` |
| fp32 | `numpy.sum` fp32 | `8.0794e-05 m` | `0.109` |
| fp32 | `torch.Tensor.sum` fp32 | `2.1325e-05 m` | `0.029` |

Três leituras obrigatórias desta tabela:

- **O resíduo não é artefato da ordem de soma.** `math.fsum`, que soma o array armazenado sem erro
  algum, dá `1.657e-13 m` — praticamente o mesmo valor. O resíduo está **nos dados**, deixado ali
  pelo arredondamento de `mean` e das subtrações. A ordem de acumulação modula esse valor em `±20%`,
  não o produz. Consequentemente, **Kahan na soma final não faria o resíduo cair**; só uma
  recentragem iterada faria, e ela é desnecessária (ver abaixo). Nenhuma soma compensada é exigida
  nem proibida.
- **A cota antiga era estruturalmente intestável em fp32.** Após conversão para fp32 o resíduo é
  `~8e-5 m`, doze ordens de grandeza acima de `1e-13 m`. A forma normalizada vale nas duas precisões
  com a mesma constante — é isso que a torna uma cota, e não a transcrição de uma medição.
- **A margem é de `~7x`** sobre o valor da semente do projeto e de `~3x` sobre o pior caso de uma
  varredura de 10 sementes (máximo observado `0.32`, semente `0`). **[M]**

**Poder discriminante.** O modo de falha que este invariante existe para pegar — recentragem
ausente, aplicada no eixo errado, ou aplicada às velocidades em vez das posições — produz
`|| sum_i r_i || ~ 1e2 m`, isto é, razão `~1e14`. A separação entre passar e falhar é de **catorze
ordens de grandeza**; qualquer limiar entre `1` e `100` é igualmente discriminante. Apertar a cota
não compra poder de detecção, só fragilidade a plataforma, versão de BLAS e ordem de redução.

**Irrelevância física, dita com número.** `1.6e-13 m` é `2.6e-14 R_0`, e `~2.9e-12` da menor escala
que o estudo alega resolver (`eps = 0.05 m`, Seção 4.3). O deslocamento de centro de massa
correspondente é `1.6e-16 m`, contra uma velocidade característica de `3.28 m/s`: ao longo de
`RUN_COLLAPSE` inteiro (`6.3 s`) ele não move nada. Este invariante testa o **amostrador**, não a
física.

**Validade:** qualquer `N`, qualquer precisão, qualquer ordem de redução. Aplicável ao estado
inicial apenas — em `t > 0` o objeto conservado é `P`, não `sum_i r_i`, e é `INV-2` que o cobre.

---

## 6. Trajetória de referência e tolerâncias

### 6.1 Construção da trajetória de referência

Não há forma fechada para `N = 1000`. A referência é numérica, construída assim:

1. **Integrador de referência:** `rk4` em **fp64**, `dt_ref = T_conv / 16384`.
2. **Validação da referência por auto-convergência:** integrar também com `dt = T_conv / 8192` e
   medir a diferença. Requisito: a diferença deve ser **pelo menos 50 vezes menor** que o menor erro
   que se pretende medir com a referência. Se falhar, refinar `dt_ref` por fator 2 e repetir.
3. A referência é gravada em `data/reference_conv_N1000.npz` (posições e velocidades em `T_conv`,
   fp64), com SHA-256 registrado. Regenerá-la é responsabilidade de um script versionado, não do
   teste.

Ratio de segurança: o ponto de teste mais fino de RK4 usa `dt = T_conv/1024`; a referência é `16×`
mais fina, logo `16⁴ = 65536` vezes mais precisa. Adequado.

**Métrica de erro** (usada em todo o documento):

```
err(r) = || r - r_ref ||_F / ( sqrt(N) * R_0 )
```

isto é, o desvio quadrático médio por partícula, normalizado pelo raio inicial da esfera.
Adimensional. Nenhuma outra normalização é aceita, para que os números deste documento sejam
diretamente comparáveis.

### 6.2 O que limita a comparação: divergência exponencial

O colapso frio é caótico. Uma perturbação do tamanho do arredondamento cresce exponencialmente.
Medido nesta configuração (`RUN_COLLAPSE`, Verlet fp64) **[M]**:

| `t / t_ff` | perturbação de 1 ULP em uma coordenada (fp64)¹ | fp32 contra fp64¹ | duas ordens de redução, ambas fp64² |
|---|---|---|---|
| `0.5` | `~7e-20` | `1.23e-06` | `1.6e-16` |
| `1.0` | `6.4e-17` | `2.35e-05` | `4.8e-15` |
| `1.5` | `7.5e-15` | `2.14e-03` | `3.9e-13` |
| `2.0` | `6.0e-13` | `4.40e-02` | `2.5e-11` |
| `3.0` | `3.35e-09` | `1.81e-01` | `2.01e-07` |

¹ medido com `dt = 1e-3`. ² medido com `dt = 5e-4` (o `dt` de `RUN_COLLAPSE`), somando a mesma
matriz de termos em 1 bloco contra 7 blocos em ordem invertida. Todas as colunas usam a métrica
`err` da Seção 6.1.

Tempo de Lyapunov ajustado no trecho pós-colapso: `t_lambda ≈ 0.25 s ≈ 0.12 t_ff`. **[M]**

Consequências, **normativas**:

1. **Comparação de trajetória entre implementações só é significativa para `t <= 0.5 t_ff`.**
2. **Comparação de trajetória fp32 contra fp64 é inválida para `t > 1.0 t_ff`.** Em `3 t_ff` o
   desvio é `0.18 R_0` — as trajetórias estão descorrelacionadas. Qualquer teste que exija
   concordância de trajetória entre precisões no fim do colapso está errado por construção e não
   deve ser escrito.
3. Para `t > 1 t_ff` só são comparáveis **quantidades conservadas** e **observáveis estatísticos
   agregados** (`r_half(t)`, `t_collapse`, `-2K/U`), que são robustos porque não dependem da
   identidade individual das partículas.

### 6.3 Tolerâncias numéricas

Base de raciocínio, medida e não estimada **[M]**:

- Erro relativo do campo de acelerações contra `float128`, na condição inicial, `N = 1000`:

  | precisão | mediana | p99 | máximo |
  |---|---|---|---|
  | fp64 | `7.0e-16` | `2.4e-15` | `3.5e-15` |
  | fp32 | `4.3e-07` | `2.0e-06` | `7.5e-06` |

- Número de condição da soma de forças, `kappa_i = sum_j |termos| / |soma|`: mediana `2.2`,
  p90 `5.0`, máximo `60` (norma vetorial). O cancelamento é modesto porque a esfera é grande frente
  a `eps`; o amplificador de arredondamento é `~60`, não `~N`.
- `eps_fp32 = 1.19e-7`, `eps_fp64 = 2.22e-16`.

Tolerâncias fixadas:

| identificador | grandeza | fp64 | fp32 | origem |
|---|---|---|---|---|
| `TOL-ACCEL` | erro relativo do campo `a` numa única avaliação, contra referência de maior precisão | `1e-13` | `5e-5` | p99 medido × margem ~40 |
| `TOL-ENERGY` | erro relativo de `U` ou `E` numa única avaliação | `1e-13` | `1e-5` | ver 6.4 |
| `TOL-MOM` | `\|sum_i a_i\| / ( N * eps_prec * max_i \|a_i\| )` | `<= 1` | `<= 1` | ver `INV-2` |
| `TOL-IMPL-SHORT` | `err(r)` entre duas implementações, mesma precisão, em `t <= 0.5 t_ff` | `1e-13` | `1e-4` | fp64 medido `1.6e-16` × margem `600`; fp32 extrapolado do fp32-vs-fp64 (`1.2e-6`) × margem `80` **[A]** |
| `TOL-IMPL-FULL` | `err(r)` entre duas implementações, mesma precisão, em `t = 3 t_ff` | `1e-5` | **não testável** | medido `2.0e-7` (fp64) × margem 50 |
| `TOL-XPREC` | `err(r)` entre fp32 e fp64, em `t <= 0.5 t_ff` | — | `1e-4` | medido `1.2e-6` × margem 80 |
| `TOL-GRAD` | erro relativo de `a` contra `-∇U/m` por diferença central `h = 1e-5 m` | `1e-8` | não aplicável | medido `3.2e-10` |
| `TOL-CENTER` | `\|\| sum_i r_i \|\| / ( N * eps_prec * R_0 )` na condição inicial | `<= 1` | `<= 1` | ver 5.5 |
| `TOL-EPI` | `A_meas / A_epi` na órbita circular suavizada, `A_epi = (omega_circ*dt)²/(4*gamma)` | `[0.99, 1.03]` | `<= 1.03` (só superior) | ver `INV-6` |

Notas normativas:

- `TOL-IMPL-FULL` para fp32 **não existe deliberadamente**. Ver 6.2, item 2.
- Toda tolerância acima é sobre grandezas **relativas e adimensionais**. Nenhum teste deve comparar
  valores absolutos em joules ou metros sem normalizar. **Esta regra vale sem exceção, inclusive
  para os valores de referência da Seção 5.4**: a cota de recentragem, que até a revisão atual era
  dada em metros, foi reescrita na forma normalizada `TOL-CENTER` (Seção 5.5) precisamente por
  violá-la. Um valor absoluto numa tabela de referência é uma medição, não uma tolerância; se for
  usado como critério de teste, o documento está errado, não o ambiente.
- Comparações CPU × GPU na **mesma** precisão caem em `TOL-IMPL-*`. A ordem de redução difere entre
  dispositivos; exigir igualdade binária é errado.

### 6.4 Diretiva sobre o diagnóstico de energia em fp32

Medido **[M]**, na condição inicial:

| forma de calcular `U` | erro relativo |
|---|---|
| termos fp32, acumulação fp32 | `8.5e-8` |
| termos fp32, acumulação fp64 | `6.2e-10` |
| termos fp64, acumulação fp64 | `7.9e-17` |

Com acumulação fp32 o piso de ruído de `U` é `~1e-7` relativo, o que **destrói** a medição de
`ΔE/E₀ ~ 1e-6` que o estudo precisa fazer.

**Diretiva:** o diagnóstico de energia (`K`, `U`, `E`) é **sempre** acumulado em fp64, mesmo quando o
núcleo dinâmico roda em fp32. Os termos de par podem ser calculados na precisão do núcleo; a
redução é fp64. Isto é uma escolha de **instrumentação**, não afeta a dinâmica, e deve estar
documentada em toda tabela de energia produzida em fp32.

---

## 7. Invariantes testáveis

Cada invariante traz: enunciado, condição de validade, tolerância e o que sua falha significa.

### `INV-1` — Consistência força/potencial

**Enunciado:** `a_i = -(1/m_i) * ∂U/∂r_i` para todo `i`.

**Procedimento:** configuração aleatória pequena (`n = 12` partículas, `rng = default_rng(7)`,
`normal(scale=2.0, size=(12,3))`), `eps = 0.05`. Diferença central com `h = 1e-5 m` em cada uma das
`3n` coordenadas.

**Tolerância:** `TOL-GRAD = 1e-8` (máximo do erro relativo sobre todas as componentes; medido
`3.2e-10`). Executar também com `h = 1e-4` e exigir `<= 1e-7`; o mínimo do erro fica em
`h ≈ 1e-5` (truncamento `O(h²)` contra arredondamento `O(eps_fp64/h)`).

**Se falhar:** o expoente do softening no potencial e na força divergiram (erro clássico:
`(dsq)^(3/2)` na força e `sqrt(dsq)` sem `eps²` no potencial, ou vice-versa). Toda a conservação de
energia do projeto fica sem sentido. **Bloqueante.**

**Validade:** qualquer `eps >= 0`, qualquer precisão fp64. Não executar em fp32.

### `INV-2` — Momento linear total

**Enunciado (a):** o campo de acelerações satisfaz `sum_i m_i a_i = 0` em aritmética exata, por
antissimetria de pares. **[T]**

**Enunciado (b):** em ponto flutuante isso **não** é exato. É importante entender por quê: em IEEE-754
a subtração `x_j - x_i` é exatamente o negativo de `x_i - x_j` (a negação é exata e a subtração é
arredondada de forma simétrica), e `dsq` é bit a bit idêntico para `(i,j)` e `(j,i)`; logo cada
**termo** de par é exatamente antissimétrico. O resíduo vem **apenas da ordem de acumulação** das
`N` parcelas de cada linha e da soma final. Não existe implementação vetorizada ingênua que conserve
`P` bit a bit; exigir isso num teste é errado.

**Cota adotada:**

```
|| sum_i m_i a_i ||  <=  N * eps_prec * max_i ( m_i * ||a_i|| )
```

Como as massas são iguais, `m` se cancela e a cota equivale a
`|| sum_i a_i || <= N * eps_prec * max_i ||a_i||`. Medido na condição inicial, em `m/s²` **[M]**:
fp64 `6.5e-14` contra cota `2.0e-12` (margem 30×); fp32 `1.1e-5` contra cota `1.1e-3`
(margem 100×). A cota é folgada o bastante para acumuladores
sequenciais e para redução em árvore.

**Enunciado (c) — deriva ao longo de uma execução:** todos os quatro integradores atualizam `v` por
combinações lineares de `a`, logo

```
|| P(t_n) - P(0) ||  <=  n_force(method, n_steps) * dt * N * eps_prec * max ||m a||
```

com `n_force` conforme a Seção 8.1 (inclui o termo de pré-condição). É uma cota de pior caso
totalmente coerente e grosseira por construção — o regime observado é de passeio aleatório, `sqrt`
do número de parcelas; o termo `+1` de `velocity_verlet` está bem abaixo da sua própria folga.

**Tolerância operacional em `RUN_COLLAPSE`:**
`|| P(t) || / ( M_tot * v_rms(t) ) <= 1e-12` em fp64 e `<= 1e-4` em fp32, para todo `t` amostrado,
para os quatro integradores.

Medido em `t = 3 t_ff`, fp64 **[M]**: `euler 1.36e-16`, `symplectic_euler 1.20e-15`,
`velocity_verlet 3.38e-16`, `rk4 1.66e-16`. Margem de ~3 ordens de grandeza sobre a cota.

Cota fp32 **[A]**, derivada e não medida: com resíduo por avaliação de `1.1e-5 m/s²` (Seção 6.3) e
`12600` passos de `5e-4 s`, o pior caso totalmente coerente dá
`ΔP/m ≈ 12600 * 5e-4 * 1.1e-5 = 6.9e-5 m/s`; com `v_rms ≈ 3 m/s` isso é `2.3e-5`. A cota de `1e-4`
tem margem 4× sobre o pior caso e ~500× sobre o caso realista de passeio aleatório
(`sqrt(12600)` em vez de `12600`).

**Se falhar:** a antissimetria de par foi quebrada — tipicamente um sinal invertido (`r_i - r_j` em
vez de `r_j - r_i`) ou uso de `m_i` onde deveria estar `m_j`. **Bloqueante.**

### `INV-3` — Momento angular total

**Enunciado:** `symplectic_euler` e `velocity_verlet` conservam `L` exatamente em aritmética exata;
`euler` e `rk4` **não**. Demonstrações na Seção 3. **[T]**

**Tolerância:**

| método | `|| L(t) - L(0) || / || L(0) ||` em `TWOBODY_ECC`, 20 períodos, 2000 passos/período, fp64 |
|---|---|
| `symplectic_euler` | `<= 1e-12` (medido `1.5e-14`) |
| `velocity_verlet` | `<= 1e-12` (medido `1.4e-14`) |
| `rk4` | `>= 1e-13` e `<= 1e-9` (medido `2.6e-11`) |
| `euler` | `>= 1e-2` (medido `2.4e-01`) |

Este é o teste que separa os métodos de forma mais limpa e barata. Ele **não** usa a esfera: `L(0)`
da esfera fria é `0` (todas em repouso), de modo que a normalização relativa é indefinida.

**Forma correspondente em `RUN_COLLAPSE`:** usar `|| L(t) ||` normalizado por
`M_tot * R_0 * v_char = 1e12 * 6.2035 * 3.2800 = 2.0347e13 kg m²/s`. Medido em `t = 3 t_ff`,
fp64 **[M]**:

| método | `\|\| L \|\| / (M_tot R_0 v_char)` | critério |
|---|---|---|
| `velocity_verlet` | `4.44e-16` | `<= 1e-12` |
| `symplectic_euler` | `8.49e-16` | `<= 1e-12` |
| `rk4` | `1.58e-13` | `>= 1e-15` e `<= 1e-10` |
| `euler` | `3.05e-04` | `>= 1e-6` |

Separação de 12 ordens de grandeza entre simpléticos e Euler explícito, com `rk4` no meio. Este é o
teste de correção mais informativo por unidade de custo do conjunto.

**Se falhar para os simpléticos:** a ordem das atualizações foi trocada, ou a força não é central.
**Bloqueante.**

### `INV-4` — Energia total por integrador

**Condição de validade:** as afirmações de "oscilação limitada" para os métodos simpléticos valem
enquanto `dt` for pequeno frente a **todas** as escalas de tempo do sistema (Seção 4.4). Elas são
garantias assintóticas, não incondicionais. Em `RUN_COLLAPSE` o sistema é caótico e passa por
encontros próximos: o comportamento observado tem componente de passeio aleatório. **Por isso o teste
quantitativo de energia é feito no problema de dois corpos regular (`INV-7`), e em `RUN_COLLAPSE` a
energia é verificada apenas contra cotas frouxas.**

**Valores medidos em `RUN_COLLAPSE`** (`eps = 0.05`, `dt = 5e-4`, `M = 12600`, fp64, diagnóstico da
Seção 6.4, amostrado a cada 20 passos) **[M]**:

| método | pico `\|ΔE/E₀\|` | `ΔE/E₀` final | razão pico/\|final\| | monótono |
|---|---|---|---|---|
| `euler` | `5.976e-01` | `+5.976e-01` | `1.0` | **sim** |
| `symplectic_euler` | `2.336e-02` | `-3.891e-04` | `60` | não |
| `velocity_verlet` | `2.288e-04` | `-3.802e-06` | `60` | não |
| `rk4` (mesmo `dt`) | `4.625e-08` | `-4.587e-08` | `1.008` | não |

Apesar do caos e dos encontros próximos, a distinção sobrevive intacta e é **testável**: nos dois
métodos simpléticos o valor final é ~60 vezes menor que o pico (**oscilação**); em `rk4` o pico e o
valor final coincidem em magnitude e o valor final é negativo (**deriva**); em `euler` pico e final
coincidem e o sinal é positivo e monótono (**crescimento**).

**Critérios de aceitação, fp64:**

| método | pico `\|ΔE/E₀\|` | `ΔE/E₀` final | forma |
|---|---|---|---|
| `euler` | — | `>= +0.3` | monótono crescente ao longo de todo o intervalo |
| `symplectic_euler` | `<= 1e-1` | `\|·\| <= pico/10` | não monótono |
| `velocity_verlet` | `<= 1e-3` | `\|·\| <= pico/10` | não monótono |
| `rk4` | `<= 1e-6` | `<= 0` e `\|·\| >= pico/3` | não monótono |

**Validade em fp32.** O **pico** de `|ΔE/E₀|` é dominado pelo truncamento, não pelo arredondamento,
e é reprodutível em fp32: medido `2.285e-4` em fp32 contra `2.288e-4` em fp64 para `velocity_verlet`
(`0.1%` de diferença) **[M]**. O **valor final**, em contrapartida, é uma quantidade caótica: fp32 dá
`-1.94e-6` contra `-3.80e-6` em fp64 (fator 2).

Portanto, em fp32:

- o critério de **pico** é válido para `euler`, `symplectic_euler` e `velocity_verlet`, cujos sinais
  (`6e-1`, `2.3e-2`, `2.3e-4`) estão muito acima do piso induzido pelo erro de força fp32 (`~1e-6`
  relativo, Seção 6.3);
- o critério de **pico** de `rk4` (`4.6e-8`) fica **abaixo** desse piso e **não é testável em fp32**;
- o critério de **valor final** não é testável em fp32 para nenhum método — apenas a razão
  qualitativa `final/pico` é, e com tolerância de fator 3.

### `INV-5` — Dois corpos kepleriano (`TWOBODY_KEPLER`, `eps = 0`)

**Configuração exata.** Duas partículas de massa `m = 1e9 kg`. Parâmetro gravitacional
`mu = G*(m1+m2) = 0.1334816 m³/s²`. Semi-eixo maior `a = 1.0 m`, excentricidade `e = 0.5`, início no
apoastro, plano `xy`, centro de massa na origem e em repouso.

```
r_apo   = a*(1+e)                    = 1.5 m
v_apo   = sqrt( mu/a * (1-e)/(1+e) ) = 0.2109356932 m/s      (velocidade RELATIVA)
period  = 2*pi*sqrt(a³/mu)           = 17.19765239 s

positions[0]  = ( +0.75, 0, 0 )      velocities[0]  = ( 0, +0.1054678466, 0 )
positions[1]  = ( -0.75, 0, 0 )      velocities[1]  = ( 0, -0.1054678466, 0 )
```

Valores exatos derivados: `r_peri = 0.5 m`, `v_rel_peri = 0.6328070796 m/s`,
`E_exact = -G m²/(2a) = -3.33704e7 J`, `||L|| = (m/2) * r_apo * v_apo = 1.582017699e8 kg m²/s`.

**Requisito de `eps`:** este teste roda com `eps = 0.0` **exatamente**. Com `eps = 0.05` a órbita
**não** é kepleriana e o teste é inválido. Ver Seção 2.5 sobre a exclusão do termo `i = j`.

**Teste:** integrar 1 período com `velocity_verlet` e verificar o retorno ao ponto inicial.
Medido **[M]**, fp64:

| passos/período | `max \|r - r⁰\|` | relativo a `r_apo` |
|---|---|---|
| 200 | `2.805e-3 m` | `1.87e-3` |
| 1000 | `1.125e-4 m` | `7.50e-5` |
| 5000 | `4.502e-6 m` | `3.00e-6` |
| 20000 | `2.814e-7 m` | `1.88e-7` |

Escala como `h²` (razão exata `25` entre 1000 e 5000 passos). **Tolerância de teste:** com 1000
passos/período, erro de retorno `<= 1e-4` relativo; e a razão de erros entre 1000 e 5000 passos deve
estar em `[20, 30]`.

Verificar também `E` contra `E_exact` (`TOL-ENERGY`) e `||L||` contra o valor analítico.

### `INV-6` — Dois corpos circular suavizado (`TWOBODY_CIRC`, `eps = 0.05`)

Este teste existe para verificar que o softening foi implementado **exatamente** como especificado, e
não apenas "aproximadamente".

**Solução analítica exata do problema suavizado.** Para separação constante `d`:

```
omega_circ = sqrt( mu / ( d² + eps² )^(3/2) )
```

Com `d = 1.0 m`, `mu = 0.1334816`, `eps = 0.05`:

```
omega_circ = 0.3646677991 rad/s
period     = 17.22988792 s
v_each     = omega_circ * d / 2 = 0.1823338995 m/s

positions[0]  = ( +0.5, 0, 0 )   velocities[0]  = ( 0, +0.1823338995, 0 )
positions[1]  = ( -0.5, 0, 0 )   velocities[1]  = ( 0, -0.1823338995, 0 )
```

**Poder discriminante:** com `eps = 0` o período seria `17.19765239 s`, uma diferença de `0.19%`. Um
teste sobre a constância da separação `d` ao longo de 10 períodos detecta um softening ausente, um
`eps` errado, ou `eps` aplicado só na força e não no potencial. Ver a tabela de sinais de falha ao
final desta seção.

#### `TOL-EPI` — a separação **não** é constante, e a amplitude é calculável

**Correção normativa.** Uma versão anterior desta seção fixava
`| |r_1 - r_0| - d | / d <= 1e-6` a 2000 passos/período. Esse número não foi derivado de `dt` e é
**menor que o desvio radial que o próprio método garante produzir**: o critério era inatingível por
qualquer implementação correta de `velocity_verlet`. A cota é substituída pela forma derivada abaixo.
Nenhum resultado publicável foi produzido sob a cota antiga.

**Enunciado.** `velocity_verlet` é simplético, logo conserva exatamente um hamiltoniano sombra
`H_h = H + O(h²)` e, neste problema central, conserva `L` exatamente. A órbita circular do problema
**contínuo** não é uma órbita circular do **mapa discreto**: para o mesmo `L`, o mapa tem sua própria
órbita circular de raio `R_h = d + O(h²)`. Partindo da condição inicial contínua, a trajetória
discreta executa uma **epicicloide de amplitude `O(h²)` em torno de `R_h`** — oscilação limitada, não
deriva. Como `v(0)` é puramente tangencial, `t = 0` é um extremo dessa oscilação, e o desvio máximo
em relação a `d` vale **duas** vezes a excentricidade radial.

Derivação da órbita circular discreta. Ao longo de uma órbita circular do mapa, de raio `R` e ângulo
`theta` por passo, o *drift* é a corda de comprimento `2R sin(theta/2) = h*V`, e o *kick* acumulado
entre meio-passos vale `2V sin(theta/2) = h*|a(R)|`. Eliminando `V`:

```
sin(theta/2) = Omega(R) * h / 2 ,        Omega(R) = sqrt( mu / (R² + eps²)^(3/2) )
L_h(R)       = R² * Omega(R) * sqrt( 1 - Omega(R)² h² / 4 )        [T]
```

`L_h(R) = L_cont(d) = d² * omega_circ` fixa `R_h`. Expandindo em `h` e escrevendo a elasticidade do
momento angular circular `L_cont(R) = R² Omega(R)`:

```
gamma  = d ln L_cont / d ln R |_(R=d)  =  2 - (3/2) * d² / (d² + eps²)

(R_h - d) / d  =  (omega_circ * h)² / (8 * gamma)  +  O(h⁴)

A_epi  =  max_t | |r_1 - r_0|(t) - d | / d  =  (omega_circ * h)² / (4 * gamma)  +  O(h⁴)    [T]
```

Casos-limite: `eps = 0` dá `gamma = 1/2` e `A_epi = (omega*h)²/2`, o resultado clássico de Verlet em
Kepler; `eps = 0.05`, `d = 1` dá `gamma = 0.5037406484` e `A_epi = 0.4962886 * (omega_circ*h)²`. O
desvio é **para fora** (`R_h > d`, pois `gamma > 0`): a separação nunca fica abaixo de `d`.

**Critério de aceitação `TOL-EPI`.** Com `velocity_verlet`, `n_per` períodos, `dt = P/spp`:

```
A_meas  =  max_t | |r_1 - r_0|(t) - d | / d          (amostrado ao longo de todo o percurso)
A_epi   =  (omega_circ * dt)² / (4 * gamma)
F_prec  =  4 * sqrt(n_steps) * eps_prec              (piso de arredondamento acumulado)

fp64:   0.99  <=  ( A_meas - F_prec ) / A_epi  <=  1.03
fp32:                A_meas                    <=  1.03 * A_epi + F_prec     (só cota superior)
```

Válido para `omega_circ * dt <= 0.07` (isto é, `>= 90` passos/período), que é onde os termos `O(h⁴)`
omitidos ficam abaixo de `1e-3` relativos. Em fp64 `F_prec` é irrelevante (`1.3e-13` a 20000 passos)
e existe apenas para que a mesma fórmula sirva às duas precisões.

**Verificação da fórmula** — `velocity_verlet`, 10 períodos, fp64, `eps = 0.05`, `d = 1` **[M]**:

| passos/período | `omega_circ*dt` | `A_meas` | `A_epi` (previsto) | `A_meas / A_epi` |
|---|---|---|---|---|
| 100 | `6.283e-2` | `1.957357e-3` | `1.959263e-3` | `0.99903` |
| 250 | `2.513e-2` | `3.134028e-4` | `3.134821e-4` | `0.99975` |
| 500 | `1.257e-2` | `7.835794e-5` | `7.837052e-5` | `0.99984` |
| 1000 | `6.283e-3` | `1.959230e-5` | `1.959263e-5` | `0.99998` |
| **2000** | `3.142e-3` | **`4.898113e-6`** | **`4.898158e-6`** | **`0.99999`** |
| 4000 | `1.571e-3` | `1.224530e-6` | `1.224539e-6` | `0.99999` |
| 16000 | `3.927e-4` | `7.653328e-8` | `7.653371e-8` | `0.99999` |

Cinco algarismos significativos sobre 160× em `dt`. A razão aproxima `1` **por baixo**, como exige um
truncamento `O(h⁴)` de sinal fixo, e nunca a excede.

**Limitação (oscilação, não deriva) — verificada explicitamente [M]**, `spp = 2000`: o máximo por
período ao longo de 10 períodos permanece em `4.891e-6 – 4.898e-6` (variação de `0.14%`, sem
tendência monótona), e ao longo de 40 períodos o máximo por bloco de 10 períodos vale
`4.898113e-6`, `4.898144e-6`, `4.898071e-6`, `4.898137e-6`. A frequência da epiciclo é
`kappa = omega_circ * sqrt(4 - 3 d²/(d²+eps²)) = 1.003733 * omega_circ`. **Se `A_meas` crescer com o
número de períodos, o simpletismo do integrador está quebrado e isso é um bug, não truncamento.**

**Poder discriminante sob `TOL-EPI`** — `spp = 2000`, 10 períodos, fp64, condição inicial sempre
construída com `eps = 0.05`, variando apenas o `eps` passado ao núcleo de força **[M]**:

| defeito injetado | `A_meas` | `A_meas / A_epi` |
|---|---|---|
| nenhum (`eps = 0.05`) | `4.898e-6` | `1.00` |
| softening ausente (`eps = 0`) | `7.444e-3` | `1520` |
| `eps` errado em `+3%` | `4.570e-4` | `93.3` |
| `eps` errado em `-3%` | `4.340e-4` | `88.6` |
| `eps` errado em `+1%` | `1.541e-4` | `31.5` |
| `eps` errado em `+0.5%` | `7.934e-5` | `16.2` |

A banda de `±3%` de `TOL-EPI` é **5× mais estreita** que o menor sinal de falha desta tabela. O
critério derivado é estritamente mais discriminante que a cota absoluta que substitui, e ainda
detecta um `eps` errado em `0.2%`.

**Aviso normativo:** a órbita circular é um caso **degenerado** para `symplectic_euler` — o termo
`O(h)` do hamiltoniano sombra se anula e a amplitude da oscilação de energia escala como `h²`,
imitando Verlet (medido: razão `4.0` em vez de `2.0`). **Não usar `TWOBODY_CIRC` para medir a ordem
da oscilação de energia.** Usar `TWOBODY_ECC` (`INV-7`).

### `INV-7` — Comportamento de energia por integrador (`TWOBODY_ECC`, `eps = 0.05`)

**Configuração:** idêntica a `INV-5`, mas com `eps = 0.05`. As condições iniciais permanecem as de
`INV-5` (o teste não requer que a órbita seja kepleriana; requer apenas que seja ligada, regular e
periódica). Integrar **20 períodos nominais** (`T = 20 * 17.19765239 s = 343.953 s`).

**Critérios de aceitação, fp64:**

| método | passos/período | amplitude `A = max(ΔE/E₀) - min(ΔE/E₀)` | razão `A(k)/A(2k)` | `\|ΔE/E₀\|` final vs. `A` |
|---|---|---|---|---|
| `symplectic_euler` | 1000, 2000, 4000, 8000 | `1.69e-2 … 2.12e-3` | `2.0 ± 0.15` | final `<= A/50` |
| `velocity_verlet` | 1000, 2000, 4000, 8000 | `1.01e-4 … 1.58e-6` | `4.0 ± 0.3` | final `<= A/1000` |
| `rk4` | 1000, 2000, 4000 | `6.74e-9 … 1.13e-11` | `>= 12` | final `>= A/3`, **negativo** |
| `euler` | 1000 … 8000 | — | — | final `>= +0.2`, **positivo** |

Este único teste estabelece, com números, toda a tabela da Seção 3.5:

- razão `2.0` para `symplectic_euler` e `4.0` para `velocity_verlet` confirmam as ordens 1 e 2 do
  erro de energia;
- "final `<= A/50`" e "final `<= A/1000`" confirmam **oscilação limitada sem deriva**;
- "final `>= A/3` e negativo" confirma **deriva secular** em RK4;
- "final `>= +0.2` e positivo" confirma o **crescimento ilimitado** de Euler explícito.

Não incluir `rk4` com 8000 passos/período no teste de razão: a amplitude (`6.5e-13`) já está no piso
de arredondamento fp64 e a razão degrada para `17.5`.

### `INV-8` — Ordem de convergência empírica em `RUN_CONVERGENCE`

**Procedimento:** para cada método, integrar `RUN_CONVERGENCE` com `n_steps ∈ {64, 128, 256, 512, 1024}`
(`dt = T_conv/n_steps`), medir `err(r)` contra a referência da Seção 6.1, e ajustar
`p_k = log2( err(n_k) / err(2*n_k) )`.

**Razão de refinamento:** `2`. **Níveis:** `5`.

**Onde começa o regime assintótico.** O erro é dominado pelo modo mais rápido do sistema, não pela
queda global: no intervalo `[0, 0.5 t_ff]` a frequência máxima medida é `omega_max ≈ 26 rad/s`
(Seção 4.4). O regime assintótico exige `omega_max * dt <~ 0.2`, isto é `dt <~ 7.7e-3 s`, isto é
**`n_steps >= 137`**. Abaixo disso o expoente medido é ruído.

**Duas métricas de erro.** O expoente deve ser medido separadamente em posição e em velocidade —
elas **não** coincidem para todos os métodos:

```
err_r = || r - r_ref ||_F / ( sqrt(N) * R_0 )         R_0     = 6.2035049090 m
err_v = || v - v_ref ||_F / ( sqrt(N) * v_char )      v_char  = 3.2799885 m/s
```

**Valores medidos, fp64, referência da Seção 6.1** (auto-convergência da referência: `1.34e-14`)
**[M]**:

| método | `n_steps` | `omega_max*dt` | `err_r` | `p_r` | `err_v` | `p_v` |
|---|---|---|---|---|---|---|
| `euler` | 64 | `0.427` | `1.7008e-02` | — | `1.0569e-01` | — |
| | 128 | `0.213` | `1.0679e-02` | `0.671` | `8.7863e-02` | `0.267` |
| | 256 | `0.107` | `6.8616e-03` | `0.638` | `6.6255e-02` | `0.407` |
| | 512 | `0.053` | `3.9202e-03` | `0.808` | `4.4113e-02` | `0.587` |
| | 1024 | `0.027` | `2.1906e-03` | `0.840` | `2.7439e-02` | `0.685` |
| | 2048 | `0.013` | `1.1526e-03` | `0.926` | `1.7548e-02` | `0.645` |
| | 4096 | `0.007` | `5.3372e-04` | `1.111` | `9.3841e-03` | `0.903` |
| `symplectic_euler` | 64 | `0.427` | `2.2221e-03` | — | `3.1687e-03` | — |
| | 128 | `0.213` | `1.1048e-03` | `1.008` | `2.6054e-04` | `3.604` |
| | 256 | `0.107` | `5.5284e-04` | `0.999` | `6.4812e-05` | `2.007` |
| | 512 | `0.053` | `2.7654e-04` | `0.999` | `1.6192e-05` | `2.001` |
| | 1024 | `0.027` | `1.3830e-04` | `1.000` | `4.0473e-06` | `2.000` |
| `velocity_verlet` | 64 | `0.427` | `2.4501e-04` | — | `3.7350e-03` | — |
| | 128 | `0.213` | `1.2465e-05` | `4.297` | `3.1691e-04` | `3.559` |
| | 256 | `0.107` | `3.1242e-06` | `1.996` | `7.9007e-05` | `2.004` |
| | 512 | `0.053` | `7.8112e-07` | `2.000` | `1.9736e-05` | `2.001` |
| | 1024 | `0.027` | `1.9528e-07` | `2.000` | `4.9329e-06` | `2.000` |
| | 2048 | `0.013` | `4.8821e-08` | `2.000` | `1.2332e-06` | `2.000` |
| `rk4` | 64 | `0.427` | `1.2163e-04` | — | `1.9602e-03` | — |
| | 128 | `0.213` | `3.7064e-06` | `5.036` | `5.8188e-05` | `5.074` |
| | 256 | `0.107` | `1.2445e-07` | `4.896` | `1.9361e-06` | `4.909` |
| | 512 | `0.053` | `3.9714e-09` | `4.970` | `6.1144e-08` | `4.985` |
| | 1024 | `0.027` | `1.2677e-10` | `4.969` | `1.9347e-09` | `4.982` |
| | 2048 | `0.013` | `4.2421e-12` | `4.901` | `6.6219e-11` | `4.869` |

**Critérios de aceitação (fp64), derivados da medição e não da teoria isolada:**

| método | `p_r` | `p_v` | razões usadas |
|---|---|---|---|
| `symplectic_euler` | `∈ [0.95, 1.05]` | `∈ [1.95, 2.10]` | a partir de `n = 256` |
| `velocity_verlet` | `∈ [1.95, 2.05]` | `∈ [1.95, 2.05]` | a partir de `n = 256` |
| `rk4` | `∈ [4.5, 5.3]` | `∈ [4.5, 5.3]` | a partir de `n = 128` |
| `euler` | `∈ [0.55, 1.20]`, crescente | `∈ [0.20, 1.10]`, tendência crescente | todas |

Quatro desvios em relação à expectativa ingênua, todos reais e todos obrigatórios de reproduzir:

1. **`velocity_verlet` na razão `64→128` dá `4.30` (posição) e `3.56` (velocidade), não `2`.** Não é
   erro: em `n = 64` temos `omega_max*dt = 0.43` e o passo não resolve o modo mais rápido. A partir
   de `n = 256` o expoente é `2.000` em ambas as métricas, sem exceção. **A primeira razão deve ser
   excluída.** O mesmo pré-assintotismo aparece em `symplectic_euler` (`p_v = 3.604` na primeira
   razão).

2. **`symplectic_euler` é de ordem 1 em posição e de ordem 2 em velocidade.** Medido: `p_r = 1.000`
   e `p_v = 2.000`, ambos limpos a 3 casas sobre três refinamentos. Este é o desvio mais perigoso do
   conjunto para o agente de testes: escrever "Euler simplético é de 1ª ordem, verificar `r` e `v`"
   produz um teste que **falha contra uma implementação correta**. A assimetria é consistente com a
   estrutura do método (a velocidade do Euler simplético é a velocidade de leapfrog avaliada em
   instantes deslocados de `h/2`, e o erro `O(h)` correspondente é um deslocamento temporal puro que
   não aparece na norma da velocidade nesta configuração) — mas essa leitura está marcada **[A]** e
   não é necessária para o teste: o critério é o valor medido.

3. **`euler` mede `p_r ≈ 0.64–1.11` e `p_v ≈ 0.27–0.90`, não `1`.** Também não é erro: o erro global
   de um método incondicionalmente instável (Seção 3.1) contém o fator de amplificação
   `(e^{L T} - 1)/L`, que só se lineariza em `dt` quando `dt` é pequeno frente ao tempo de
   amplificação. O expoente medido **cresce** rumo a `1` (posição: `0.638 → 0.808 → 0.840 → 0.926 →
   1.111`) e o erro absoluto permanece grande (`5e-4` mesmo com `4096` passos, contra `4.9e-8` de
   Verlet com `2048`). O critério correto é "tendência a `1` por baixo", não "`p = 1`".

4. **`rk4` mede `p ≈ 4.9–5.0` em posição E em velocidade, não `4`.** Resultado empírico
   **não explicado**, e é a única questão física em aberto deste documento.

   O que já foi descartado:
   - **Não** é artefato de arredondamento: em `n = 2048` o erro é `4.2e-12`, ainda `300×` acima do
     piso da referência (`1.3e-14`).
   - **Não** é artefato de referência: a referência é `8×` mais fina que o ponto mais fino testado.
   - **Não** é pré-assintotismo: o expoente é estável (`4.90–5.04`) sobre um fator `32` em `h`,
     cobrindo `omega_max*dt` de `0.43` a `0.013`, enquanto `velocity_verlet` no mesmo intervalo já
     havia assentado em `2.000` a partir de `omega_max*dt = 0.107`.
   - **Não** é a hipótese estrutural de Nyström. Ela previa `p_r = 5` com `p_v = 4` (o incremento de
     posição do RK4 clássico aplicado a `r'' = A(r)` reduz-se a `r + h v + (h²/6)(k1_v+k2_v+k3_v)`,
     cuja expansão coincide com a exata até `h⁴` inclusive). A medição dá `p_v = 4.87–5.07`.
     **Hipótese refutada.**

   Hipóteses remanescentes, nenhuma verificada **[A]**: (a) o coeficiente do termo global `O(h⁴)` é
   anomalamente pequeno nesta configuração, de modo que o termo `O(h⁵)` domina em toda a faixa
   testada; (b) os erros locais não se acumulam coerentemente sobre `[0, T_conv]`, e o que se mede é
   o erro local e não o global.

   **Diagnóstico obrigatório antes de o relatório afirmar qualquer coisa sobre a ordem de RK4:**
   repetir a escada com `T = 0.25 t_ff` e com `T = 1.0 t_ff`, mantendo tudo o mais fixo. As duas
   hipóteses fazem previsões opostas — sob (a) o expoente permanece `≈ 5` em qualquer `T`; sob (b)
   ele cai rumo a `4` conforme `T` cresce e mais passos contribuem.

   **Instrução ao agente de testes:** aceitar `p ∈ [4.5, 5.3]` em ambas as métricas. Não escrever
   teste que exija `p ≈ 4`.

   **Instrução ao redator:** o relatório pode afirmar "RK4 é de 4ª ordem" apenas como propriedade
   **do método**; ao descrever a medição deve reportar o expoente medido e registrar a discrepância
   como investigação em aberto. **Não arredondar `4.95` para `4` na prosa.**

**Executar apenas em fp64.** Em fp32 o piso de arredondamento (`err ~ 1e-6`, Seção 6.2) é atingido
por `velocity_verlet` já em `n_steps ~ 256` e por `rk4` em `n_steps ~ 64`; a ordem medida em fp32 é
uma medida do arredondamento, não do método. Isto pode ser mostrado como resultado, mas não como
teste de correção.

**Se falhar (`p` sistematicamente uma unidade abaixo do valor da tabela):** acoplamento cruzado
errado nos estágios de RK4 (Seção 3.4), ou fusão indevida dos laços em `symplectic_euler`
(Seção 3.2), ou ausência do reaproveitamento de `a` em `velocity_verlet` (Seção 3.3).

### `INV-9` — Tempo de colapso contra `t_ff`

**Definição operacional de `t_collapse`.** Sobre a grade de saída de passo `OUT_DT = 1e-2 s`
(independente de `dt`; `dt` deve dividir `OUT_DT`):

1. Em cada instante de saída, calcular o centro de massa `c(t) = (1/N) * sum_i r_i` e as distâncias
   `d_i = |r_i - c|`.
2. `r_half(t) := ` a `(N/2)`-ésima menor distância (índice `N//2 - 1` no vetor ordenado, base 0).
3. `k* := argmin_k r_half(t_k)`.
4. Refinar por parábola nos três pontos `(t_{k*-1}, t_{k*}, t_{k*+1})`:
   `t_collapse = t_{k*} + (OUT_DT/2) * (y_{-1} - y_{+1}) / (y_{-1} - 2*y_0 + y_{+1})`.

**Valores de referência, `RUN_COLLAPSE`, `velocity_verlet` fp64** **[M]**:

| grandeza | valor |
|---|---|
| `r_half(0)` | `4.881251 m` |
| `r_half,min` | `0.3472 m` |
| fator de contração `r_half(0)/r_half,min` | `14.1` |
| `t_collapse` (grade) | `2.180 s` |
| `t_collapse` (parábola) | `2.1766 s` |
| `t_collapse / t_ff` | **`1.0361`** |

Robustez verificada **[M]**: `t_collapse/t_ff = 1.0361` idêntico a 4 casas para
`dt ∈ {2e-3, 1e-3, 5e-4, 2.5e-4}`; e `1.0346–1.0371` sobre `eps ∈ {0.02, 0.05, 0.1}` (tabela da
Seção 4.3). `r_half,min` é menos robusto: `0.344 – 0.360 m` sobre a mesma varredura (`4.5%`).

**Critério de aceitação:** `| t_collapse / t_ff - 1 | <= 0.10`.

A margem de `10%` é deliberadamente maior que o desvio medido de `3.6%`. O excesso de `3.6%` sobre o
valor analítico é **físico**, não numérico: `t_ff` é o resultado do limite contínuo com colapso
homólogo exato, enquanto uma realização de Poisson com `N = 1000` tem ruído de densidade, cruzamento
de camadas e pressão efetiva de discreteza que atrasam o mínimo. **Não "corrigir" essa discrepância.**
A tolerância de `10%` continua sendo um teste forte: um erro de fator 2 em `G`, `m`, `R_0`, ou um
expoente errado na lei de força, produzem desvios de dezenas de por cento a ordens de grandeza.

Critério secundário (mais frouxo, informativo): `r_half,min ∈ [0.30, 0.40] m`.

**Validade em fp32.** `t_collapse` e `r_half,min` são observáveis **agregados** (não dependem da
identidade individual das partículas) e por isso sobrevivem à divergência caótica. Medido **[M]**,
Verlet, `RUN_COLLAPSE`:

| precisão | `r_half,min` | `t_collapse` (grade) | `t_collapse/t_ff` (parábola) | pico `\|ΔE/E₀\|` |
|---|---|---|---|---|
| fp64 | `0.3472 m` | `2.180 s` | `1.0361` | `2.288e-4` |
| fp32 | `0.3473 m` | `2.180 s` | `1.0361` | `2.285e-4` |

Concordância de 4 casas decimais em `t_collapse` e de `0.03%` em `r_half,min`, apesar de as
trajetórias individuais estarem descorrelacionadas (`err ≈ 0.18` em `3 t_ff`, Seção 6.2). **Este é o
teste que valida a implementação fp32**, no lugar da comparação de trajetória, que é inválida.

**Critério fp32:** `|t_collapse(fp32) - t_collapse(fp64)| <= 0.02 * t_ff` e
`|r_half,min(fp32)/r_half,min(fp64) - 1| <= 0.02`.

### `INV-10` — Limitação inferior da energia potencial

`U(t) >= -G m² N(N-1)/(2 eps) = -6.6674e14 J` para todo `t`, por construção do núcleo de Plummer.
**[T]** Violação indica `eps` não aplicado no potencial. Teste barato, executar em todo passo
amostrado, ambas as precisões.

---

## 8. Protocolo de comparação justa entre integradores

### 8.1 O eixo é o número de avaliações de força

RK4 custa `4` avaliações de força por passo; os outros três custam `1`. Comparar erro contra número
de passos favorece RK4 por um fator `4` de custo oculto. O eixo de comparação é

```
n_force(method, n_steps) = n_steps * FORCE_EVALS_PER_STEP[method] + FORCE_EVALS_STARTUP[method]
```

com as duas tabelas conforme a Seção 3.5 (valores na Seção 9). `n_force` é a métrica de custo
**normativa**: em `O(N²)` por avaliação, ela é proporcional ao trabalho aritmético e independe de
hardware, precisão e linguagem — ao contrário do tempo de parede, que é o objeto do benchmark, não da
física.

**O termo de pré-condição entra no eixo. Ele não é amortizado nem descontado.** **[T]**

A alternativa — declarar a avaliação `a^0 = A(r^0)` de `velocity_verlet` "amortizada" e excluí-la do
eixo — foi **rejeitada**. Ela é aritmética real, `O(N²)`, indistinguível de qualquer outra avaliação;
e "amortizada" é uma afirmação assintótica, verdadeira exatamente onde não importa (`M` grande) e
falsa onde a Seção 8.2 opera de propósito (`M` pequeno). Descontar 1 de 65 avaliações é uma versão
menor do mesmo fator oculto que o fator 4 de RK4, e esta seção existe para eliminar fatores ocultos,
não para escolher quais tolerar. Se o custo de borda for irrelevante — e ele é, ver 8.2 —, isso deve
ser **demonstrado com o número no eixo**, não obtido apagando o número.

**Diagnósticos não entram em `n_force`.** As funções de `observables` (`total_energy`,
`half_mass_radius`, …) não chamam o núcleo de força e, por contrato, não são executadas dentro da
região cronometrada. `n_force` mede a dinâmica; instrumentação é caminho separado.

### 8.2 Escadas de `dt` a custo igual

**Decisão normativa.** `B` é um **orçamento nominal**, não o custo exato. A escada de `dt` é definida
pelo orçamento; o custo real é medido pela fórmula da Seção 8.1 e **reportado como tal**, com o termo
de pré-condição incluído. Fixado `B` sobre `[0, T]`:

```
n_steps          = B / FORCE_EVALS_PER_STEP[method]
dt               = T / n_steps
n_force (real)   = n_steps * FORCE_EVALS_PER_STEP[method] + FORCE_EVALS_STARTUP[method]
```

Para `RUN_CONVERGENCE` (`T = T_conv = 1.05035175 s`), a tabela normativa é:

| `B` nominal | `dt` de `euler`, `symplectic_euler`, `velocity_verlet` | `n_steps` | `n_force` real de `euler` / `symplectic_euler` | `n_force` real de `velocity_verlet` | `dt` de `rk4` | `n_steps` de `rk4` | `n_force` real de `rk4` |
|---|---|---|---|---|---|---|---|
| 256 | `4.10294e-3` | 256 | 256 | **257** | `1.641174e-2` | 64 | 256 |
| 512 | `2.05147e-3` | 512 | 512 | **513** | `8.205873e-3` | 128 | 512 |
| 1024 | `1.025734e-3` | 1024 | 1024 | **1025** | `4.102936e-3` | 256 | 1024 |
| 2048 | `5.128670e-4` | 2048 | 2048 | **2049** | `2.051468e-3` | 512 | 2048 |
| 4096 | `2.564335e-4` | 4096 | 4096 | **4097** | `1.025734e-3` | 1024 | 4096 |
| 8192 | `1.282167e-4` | 8192 | 8192 | **8193** | `5.128670e-4` | 2048 | 8192 |

`B` deve ser múltiplo de 4. Os valores de `dt` são **inalterados** em relação à revisão anterior
desta tabela: a decisão corrige o eixo de custo, não a escada de passos.

**Por que a escada de `dt` não foi reajustada para custo exatamente igual.** A alternativa seria
`n_steps = B - 1` para `velocity_verlet` (255, 511, 1023, …), de modo que `n_force` casasse com `B`
exatamente. **Rejeitada**, por três razões, em ordem de peso:

1. **Destruiria o reaproveitamento dos dados de `INV-8`.** Esta seção declara logo abaixo que o eixo
   `n_force` é obtido **reprocessando as mesmas execuções** da escada de convergência. Com
   `n_steps ∈ {255, 511, 1023}` nenhum ponto de `INV-8` serviria, e `velocity_verlet` exigiria uma
   campanha inteira e exclusiva de execuções. É um custo real, pago para corrigir um erro que não é
   real.
2. **Corromperia a razão de refinamento.** `511/255 = 2.0039`, não `2`. O estimador
   `p_k = log2(err(n_k)/err(2 n_k))` de `INV-8` pressupõe razão exatamente `2`; qualquer confusão
   entre as duas escadas passaria a introduzir viés no expoente. Manter as duas escadas sobre
   potências de dois é o que torna a distinção entre elas segura.
3. **A grandeza corrigida é menor que a espessura da linha do gráfico.** O excesso relativo de
   `velocity_verlet` é `1/n_steps`, isto é `0.391%` no ponto mais grosseiro desta tabela
   (`B = 256`), caindo a `0.012%` em `B = 8192`. No eixo logarítmico da Seção 8.3 isso são
   **`0.0017` décadas** no pior caso — contra diferenças verticais entre métodos de 3 a 6 décadas.
   **[M]**

   Quando os pontos de `INV-8` com `n_steps = 64` forem reprocessados sobre o eixo `n_force`, o
   excesso sobe para `1/64 = 1.56%`, ou `0.0067` décadas. Ainda invisível, e — ponto essencial — sem
   nenhum efeito sobre a **ordenada** desses pontos: `err_r(n_steps = 64) = 2.4501e-04` para
   `velocity_verlet` é o mesmo número antes e depois desta decisão. O pré-assintotismo documentado
   em `INV-8`, ressalva 1 (`p_r = 4.297` na razão `64 -> 128`) é um fenômeno de `dt` e **não tem
   relação alguma** com a contabilidade de custo. Não o atribua a ela.

Verificação de estabilidade (inalterada, porque `dt` é inalterado): o maior `dt` da tabela é
`1.64e-2 s` (RK4, `B = 256`); com `omega_max = 42 rad/s`, `omega*dt = 0.69`, dentro do limite `2.828`
de RK4. Os métodos de 1 avaliação nunca ultrapassam `dt = 4.1e-3`, isto é `omega*dt = 0.17`, muito
abaixo do limite `2`. Nenhum ponto da tabela é instável. **[T]**

**Obrigação de reporte.** Todo artefato de saída que carregue um eixo de custo (CSV, legenda,
gráfico) deve gravar o `n_force` **real** por execução, calculado pela fórmula da Seção 8.1, e não o
`B` nominal. Se as duas colunas existirem, devem estar ambas presentes e nomeadas distintamente
(`budget` e `n_force`). Nenhum gráfico de erro contra `n_force` pode ser produzido antes de a
implementação expor `FORCE_EVALS_STARTUP`.

**Relação com a escada de `INV-8`.** São duas escadas com propósitos diferentes e não devem ser
confundidas:

- `INV-8` (ordem de convergência) usa `n_steps ∈ {64, 128, 256, 512, 1024}` **por método**, cada
  método com o mesmo número de *passos*. É a única forma de medir o expoente `p`, que é definido
  contra `dt`.
- Esta seção (comparação justa) usa o mesmo `B` de *avaliações de força* para todos os métodos. É a
  única forma de responder "qual método entrega mais precisão pelo mesmo custo".

O eixo `n_force` da Seção 8.3 é obtido reprocessando os mesmos dados: um ponto medido com
`n_steps = k` aparece em `n_force = k * FORCE_EVALS_PER_STEP[m] + FORCE_EVALS_STARTUP[m]` — ou seja,
em `k` para `euler` e `symplectic_euler`, em `k + 1` para `velocity_verlet` e em `4k` para `rk4`. Os
pontos de `B = 8192` exigem
`n_steps = 8192` para os métodos de 1 avaliação, fora da escada de `INV-8`; são execuções adicionais
e a referência da Seção 6.1 deve ser validada contra o erro esperado nesse ponto antes de usá-los.

### 8.3 O gráfico normativo

Eixo `x`: `n_force` (log). Eixo `y`: `err(r)` em `t = T_conv` (log). Uma curva por método, quatro
curvas. As inclinações assintóticas **medidas** (Seção `INV-8`) são `-1.11` (euler, ainda subindo
rumo a `-1` por baixo), `-1.00` (symplectic_euler), `-2.00` (velocity_verlet) e `-4.95` (rk4) — não
`-4` para RK4; ver a ressalva 4 de `INV-8`. Os **interceptos** — e portanto quem vence a que orçamento — são o
resultado do estudo e não podem ser antecipados aqui.

Cuidado ao ler o gráfico: o deslocamento de RK4 no eixo `n_force` é de `log2(4) = 2` posições à
direita em relação ao gráfico contra `n_steps`. É exatamente esse deslocamento que a Seção 8.1
existe para impor.

O deslocamento correspondente de `velocity_verlet` (`+1` avaliação, Seção 3.3) é de `log2(1 + 1/M)`
posições, isto é `0.0224` posições em `M = 64` e menos daí em diante — abaixo da resolução gráfica.
**Ele deve estar nos dados de qualquer forma**, e a legenda deve declarar que o eixo inclui a
avaliação de pré-condição. O eixo é honesto ou não é um eixo de custo; que o efeito seja invisível é
um **resultado**, não uma licença para omiti-lo.

Um segundo gráfico, `|ΔE/E₀|` contra `n_force` sobre `TWOBODY_ECC` (`INV-7`), separa simpléticos de
não simpléticos e é o argumento que sustenta a recomendação final.

### 8.4 O que não pode ser afirmado

- Nada sobre trajetórias individuais em `t > 1 t_ff` comparando precisões ou dispositivos (6.2).
- Nada sobre "ordem de convergência" medida em fp32 (`INV-8`).
- Nada sobre conservação de energia a partir de `RUN_BENCH` (`M = 1`).
- Nada sobre estrutura do núcleo colapsado em escalas `< eps = 0.05 m` (4.3, item 2).
- Nada sobre relaxação de dois corpos ou evaporação: em `3 t_ff` o sistema percorreu `~3` tempos de
  cruzamento, contra um tempo de relaxação de `~0.1 N / ln N ≈ 14` tempos de cruzamento. O sistema
  **não** está relaxado ao fim de `RUN_COLLAPSE`. **[A]** — estimativa padrão, não medida aqui.

---

## 9. Constantes para a implementação

```
G                 = 6.67408e-11        # m^3 kg^-1 s^-2
PARTICLE_MASS     = 1.0e9              # kg
N_PARTICLES       = 1000
SPHERE_RADIUS     = 6.2035049090       # m   = (3N/(4pi))^(1/3)
SOFTENING         = 5.0e-2             # m
SEED              = 20190222

T_FF              = 2.1007035          # s   = sqrt(3 pi/(32 G rho)), rho = 1e9
T_CONV            = 1.05035175         # s   = 0.5 * T_FF

# escalas derivadas (para normalizacoes de teste)
TOTAL_MASS        = 1.0e12             # kg  = N * PARTICLE_MASS
DENSITY           = 1.0e9              # kg/m^3
V_CHAR            = 3.2799885          # m/s = sqrt(G*TOTAL_MASS/SPHERE_RADIUS)
L_SCALE           = 2.0347400e13       # kg m^2/s = TOTAL_MASS*SPHERE_RADIUS*V_CHAR
OMEGA_PAIR_MAX    = 32.678             # rad/s = sqrt(G*2*PARTICLE_MASS/SOFTENING^3)
P_EPS             = 0.192276           # s   = 2 pi / OMEGA_PAIR_MAX
OMEGA_MAX_DESIGN  = 42.0               # rad/s (maximo medido do tensor de mare em RUN_COLLAPSE)
U_MIN_BOUND       = -6.6674e14         # J   = -G m^2 N(N-1)/(2 eps)

# RUN_COLLAPSE
DT_COLLAPSE       = 5.0e-4             # s
N_STEPS_COLLAPSE  = 12600
OUT_DT            = 1.0e-2             # s

# RUN_CONVERGENCE
CONV_N_STEPS      = (64, 128, 256, 512, 1024, 2048)   # + 4096 para euler
REF_N_STEPS       = 16384              # rk4, fp64
REF_CHECK_N_STEPS = 8192

# contabilidade de custo (Secoes 3.5, 8.1, 8.2)
# n_force(method, M) = M * FORCE_EVALS_PER_STEP[method] + FORCE_EVALS_STARTUP[method]
FORCE_EVALS_PER_STEP = {"euler": 1, "symplectic_euler": 1, "velocity_verlet": 1, "rk4": 4}
FORCE_EVALS_STARTUP  = {"euler": 0, "symplectic_euler": 0, "velocity_verlet": 1, "rk4": 0}

# tolerancia de recentragem da condicao inicial (Secao 5.5)
# TOL-CENTER: ||sum_i r_i|| / (N * eps_prec * SPHERE_RADIUS) <= TOL_CENTER
TOL_CENTER        = 1.0                # adimensional, fp64 e fp32
EPS_FP64          = 2.220446049250313e-16
EPS_FP32          = 1.1920929e-07

# RUN_BENCH (herdado de 2019, sem afirmação física)
DT_BENCH          = 1.0e-2             # s
N_STEPS_BENCH     = 1

# dois corpos
TWOBODY_MU        = 0.1334816          # m^3/s^2 = G*(m1+m2)
KEPLER_A          = 1.0                # m
KEPLER_E          = 0.5
KEPLER_PERIOD     = 17.19765239        # s
CIRC_SEPARATION   = 1.0                # m
CIRC_OMEGA        = 0.3646677991       # rad/s  (eps = 0.05)
CIRC_PERIOD       = 17.22988792        # s

# valores de referencia medidos (fp64) -- ver Secoes 5.4, INV-4, INV-8, INV-9
IC_R_HALF_0       = 4.881251           # m
IC_R_MAX          = 6.323302           # m
IC_U_EPS005       = -6.4260397026e12   # J
IC_A_MAX          = 9.075891           # m/s^2
IC_A_RMS          = 1.544392           # m/s^2
COLLAPSE_R_HALF_MIN   = 0.3472         # m
COLLAPSE_T_OVER_TFF   = 1.0361         # adimensional
```

---

## 10. Resumo das decisões que este documento fixa

1. O softening de 2019 **é** Plummer; a forma funcional é mantida, o valor muda de `1e-10 m`
   (guarda de divisão por zero, não física) para `5e-2 m` (regularização física que limita
   `omega_max` e torna `dt` fixo defensável).
2. O esquema de 2019 **é** Euler semi-implícito (simplético), variante velocidade primeiro.
   Confirmado nos quatro núcleos do notebook.
3. A esfera fria usa `R_0 = 6.2035 m` porque isso reproduz exatamente o volume, a massa e a
   densidade da grade cúbica de 2019 — mesmo regime dinâmico, custo idêntico.
4. `dt = 5e-4 s` e `M = 12600` para o colapso; `dt = 0.01` de 2019 é inadmissível para física
   (15 passos por período do modo mais rápido) e sobrevive apenas no benchmark de tempo com `M = 1`.
5. O sistema simulado **não** é kepleriano. O teste de Kepler roda com `eps = 0`; o teste com
   softening usa a órbita circular suavizada, que tem forma fechada própria.
6. Trajetórias só são comparáveis entre implementações até `0.5 t_ff` (fp32) e `3 t_ff` com
   tolerância `1e-5` (fp64). O tempo de Lyapunov medido é `~0.12 t_ff`.
7. O eixo de comparação entre integradores é `n_force`, não `n_steps`, e `n_force` inclui o termo de
   pré-condição: `n_force = M * FORCE_EVALS_PER_STEP + FORCE_EVALS_STARTUP`. Apenas
   `velocity_verlet` tem `FORCE_EVALS_STARTUP = 1` (a semeadura `a^0 = A(r^0)` da cadeia de
   reaproveitamento); `euler`, `symplectic_euler` e `rk4` têm `0` — RK4 clássico não é FSAL, logo
   custa exatamente `4M`. As escadas de `dt` da Seção 8.2 **não** foram reajustadas: o efeito é de
   `0.0017` a `0.0067` décadas no eixo logarítmico, e reajustá-las quebraria o reaproveitamento dos
   dados de `INV-8` e a razão de refinamento exata `2`. O custo real é reportado; a escada é
   preservada.
8. A validação em fp32 é feita por **observáveis agregados** (`t_collapse`, `r_half,min`, pico de
   `|ΔE/E₀|`), que concordam com fp64 dentro de `0.1%`, e **não** por comparação de trajetória, que é
   inválida além de `0.5 t_ff`.
9. Quatro expoentes medidos divergem da expectativa ingênua e estão documentados com o valor medido,
   não com o valor de livro (`INV-8`, ressalvas 1 a 4): `velocity_verlet` na razão mais grosseira
   (`4.30`, pré-assintótico); **`symplectic_euler` é de ordem 1 em posição mas de ordem 2 em
   velocidade** (`1.000` / `2.000`, limpo a 3 casas); `euler` (`0.64 → 1.11`, tendendo a `1` por
   baixo); e `rk4` (`4.95` em posição **e** em velocidade, em vez de `4`).
10. A única questão física **em aberto** é a ordem medida de RK4. Descartados arredondamento,
    referência, pré-assintotismo e a hipótese estrutural de Nyström. O diagnóstico que decide entre
    as duas hipóteses restantes está especificado em `INV-8`, ressalva 4, e é **pré-requisito** para
    qualquer afirmação sobre a ordem de RK4 no relatório final.
11. O resíduo da recentragem é testado pela forma **adimensional** `TOL-CENTER`
    (`|| sum_i r_i || / (N * eps_prec * R_0) <= 1`, Seção 5.5), e não pela cota absoluta `1e-13 m`
    da revisão anterior, que violava a própria regra da Seção 6.3 e era intestável em fp32. Medido
    `0.116` (NumPy) e `0.140` (torch) em fp64, `0.029` a `0.109` em fp32.
