# Especificação física executável — extensões estocásticas

Documento normativo. Fixa o problema contínuo, a discretização, os parâmetros, os invariantes
testáveis e as tolerâncias de três extensões à simulação de N corpos do projeto TCCAstro:
**espectro de massas**, **velocidades iniciais térmicas** e **colisões**.

**Status.** Vinculante. O agente de implementação escreve código a partir deste documento; o agente
de testes escreve testes a partir deste documento, sem ler `src/`. Toda constante, tolerância e
critério de aceitação necessários aos testes estão aqui. Se um teste precisar de um número que não
esteja neste documento, o documento está incompleto e deve ser corrigido — não o teste.

**Relação com `docs/integradores.md`.** Aquele documento permanece vinculante e **não é revogado**.
Este o estende. Onde os dois divergirem sobre a física do problema suave (equações de movimento,
softening, integradores, `RUN_COLLAPSE`, `RUN_CONVERGENCE`, `INV-1` a `INV-10`), `integradores.md`
prevalece. Este documento acrescenta `INV-11` em diante. As emendas que ele exige em documentos
existentes estão listadas na Seção 9 e **não** foram aplicadas por este documento.

**Convenção.** Texto em português; equações, identificadores e nomes de arquivo em inglês. Unidades
SI ao longo de todo o documento.

**Rastreabilidade das afirmações.** Cada afirmação quantitativa está marcada como:
- **[T]** garantido teoricamente (demonstração algébrica dada ou referenciada aqui);
- **[M]** medido nesta configuração específica (valor de referência obtido em fp64);
- **[A]** assumido, ainda não verificado — acompanhado da medição que o decidiria.

Este documento é escrito **antes** de a implementação existir. Consequentemente há aqui muito menos
**[M]** e muito mais **[A]** do que em `integradores.md`. Isso é deliberado e deve permanecer
visível: todo **[A]** traz explicitamente qual medição o converte em **[M]**. Um **[A]** que vire
número na prosa do relatório sem passar por essa medição é uma falha de processo.

---

## 1. Escopo, e o que estas extensões fazem com o resultado existente

### 1.1 As três extensões

1. **Espectro de massas.** As massas deixam de ser iguais. `dN/dm ∝ m^-alpha` truncada, com
   `1 <= k <= 3` partículas "massudas" por realização.
2. **Velocidades iniciais.** O estado inicial deixa de ser frio. Maxwelliana isotrópica truncada,
   parametrizada pela razão virial `Q = 2K/|U|`. `Q = 0` reproduz o colapso frio existente.
3. **Colisões.** Eventos discretos de contato, com três desfechos (elástica, fusão, fragmentação).

As três são independentes entre si e devem ser implementáveis e testáveis separadamente. Nenhuma
delas pode alterar o caminho de código quando desligada — ver `INV-17` e `INV-30`.

### 1.2 Aviso normativo sobre o resultado-chefe do projeto

O resultado-chefe registrado em `results/2026/longrun_energy.csv` é a **banda limitada de
`|ΔE/E₀|`** dos integradores simpléticos ao longo de 10 `t_ff` (`velocity_verlet`:
`max|ΔE/E₀| = 3.526e-05`, final `-9.402e-07`, sem cruzar a tolerância de 5%). **[M]**

As extensões 1 e 2 **preservam** esse resultado: mudar massas e velocidades iniciais muda a
trajetória, não a estrutura do integrador. O fluxo continua hamiltoniano e suave, e as garantias
`[T]` de `integradores.md` valem sem alteração. O que muda são os **valores** medidos, não os
critérios qualitativos.

A extensão 3 **destrói** esse resultado no seu enunciado atual. Colisões são eventos discretos; a
análise de erro para trás que sustenta o hamiltoniano sombra vale para o fluxo suave. A Seção 4.11
diz com precisão o que sobrevive, o que não sobrevive e qual diagnóstico substitui o antigo. Nenhum
número da campanha colisional pode ser comparado diretamente com `longrun_energy.csv`.

### 1.3 Estágios

O documento é escrito para três estágios de execução, e vários números só existem depois do estágio
que os mede:

| estágio | o que produz | o que fixa |
|---|---|---|
| 1 | espectro de massas + velocidades, sem colisões | `t_ff` realizado, `r_half,min(Q)`, `t_collapse(Q)` |
| 2 | detecção de colisão ligada, **todos** os desfechos elásticos | histograma de `x`, taxa de eventos, `C_coll` medido |
| 3 | os três desfechos, ensemble de sementes | estatística de canais, `N_final`, `t_50` |

`R_ref` e a calibração do mapa de regime (Seção 4.6) são fixados **pela medição do estágio 2**, não
por este documento. Este documento fixa a **grandeza a medir** e a **faixa-alvo**.

---

## 2. Espectro de massas

### 2.1 Distribuição alvo

Lei de potência truncada em `[m_min, m_max]`, com `alpha = 2.35` (Salpeter) por padrão e razão de
truncamento `R = m_max/m_min = 1000` fixada.

Densidade de probabilidade normalizada, para `alpha != 1`:

```
p(m) = C * m^(-alpha) ,        C = (1 - alpha) / ( m_max^(1-alpha) - m_min^(1-alpha) )
```

e, para `alpha = 1` (caso degenerado, a normalização acima é `0/0`):

```
p(m) = 1 / ( m * ln(m_max/m_min) )
```

### 2.2 CDF e CDF inversa — com o caso degenerado

Para `alpha != 1`:

```
F(m)      = ( m^(1-alpha) - m_min^(1-alpha) ) / ( m_max^(1-alpha) - m_min^(1-alpha) )
F^-1(u)   = [ m_min^(1-alpha) + u * ( m_max^(1-alpha) - m_min^(1-alpha) ) ]^( 1/(1-alpha) )
```

Para `alpha = 1`:

```
F(m)      = ln(m/m_min) / ln(m_max/m_min)
F^-1(u)   = m_min * (m_max/m_min)^u
```

**Armadilha de implementação (normativa):** escrever apenas o ramo `alpha != 1` e confiar em que
`alpha = 2.35` nunca mudará produz uma divisão por zero silenciosa no dia em que alguém varrer
`alpha`. Os dois ramos são obrigatórios, com o corte feito por comparação exata `alpha == 1.0`
**não** é aceitável — usar `|alpha - 1| < 1e-12`, e o mesmo para `alpha = 2` na Seção 2.3.

**Amostragem por CDF inversa restrita a um subintervalo.** Para amostrar de `p` condicionada a
`m ∈ [a, b] ⊆ [m_min, m_max]`, sorteia-se `u ~ U(0,1)` e calcula-se

```
m = F^-1( F(a) + u * ( F(b) - F(a) ) )
```

Isto é **exatamente** a lei de potência truncada em `[a, b]` **[T]**: a transformação integral de
probabilidade é exata, e restringir a imagem de `F` a `[F(a), F(b)]` é a definição de condicionamento.

### 2.3 Massa média em forma fechada — com os dois casos degenerados

```
alpha != 1 e alpha != 2:
    <m> = ( (1-alpha) / (2-alpha) ) * ( m_max^(2-alpha) - m_min^(2-alpha) )
                                    / ( m_max^(1-alpha) - m_min^(1-alpha) )

alpha = 2:
    <m> = ln(m_max/m_min) / ( 1/m_min - 1/m_max )

alpha = 1:
    <m> = ( m_max - m_min ) / ln(m_max/m_min)
```

Os dois casos degenerados são independentes: `alpha = 1` anula o denominador da normalização,
`alpha = 2` anula o expoente `2-alpha` da integral de massa. Uma implementação que trate só um dos
dois está errada pela metade.

### 2.4 Resolução de `m_min` — não é numérica

**Correção normativa.** O enunciado do projeto pede `m_min` "resolvido numericamente" para que
`<m> = PARTICLE_MASS`. Com a razão de truncamento `R = m_max/m_min` **fixada**, isso é desnecessário:
substituindo `m_max = R * m_min` nas expressões acima, `m_min` sai por fator comum e

```
<m> = m_min * g(alpha, R)

g(alpha, R) = ( (1-alpha)/(2-alpha) ) * ( R^(2-alpha) - 1 ) / ( R^(1-alpha) - 1 )      [alpha != 1, 2]
g(alpha, R) = ln(R) / ( 1 - 1/R )                                                       [alpha = 2]
g(alpha, R) = ( R - 1 ) / ln(R)                                                         [alpha = 1]

  =>   m_min = PARTICLE_MASS / g(alpha, R)          [T], forma fechada
```

`g` é adimensional e depende só de `(alpha, R)`. **Não há raiz a buscar.** Busca numérica só seria
necessária se `m_max` fosse fixado independentemente de `m_min`, que não é o caso aqui.

Valores para `alpha = 2.35`, `R = 1000`, `PARTICLE_MASS = 1e9 kg` **[T]** (aritmética de 40 dígitos):

| grandeza | valor |
|---|---|
| `g(2.35, 1000)` | `3.5136877959` |
| `m_min` | `2.8460126741e8 kg` |
| `m_max` | `2.8460126741e11 kg` |
| `<m>` (verificação) | `1.0000000000e9 kg` |

**Requisito de teste:** a implementação deve expor `g` (ou `m_min`) e o teste deve verificar
`<m> = PARTICLE_MASS` **por quadratura numérica independente da fórmula fechada**, não reusando a
mesma expressão. Fórmula fechada verificada contra fórmula fechada não é teste.

### 2.5 O limiar `m_big` e a construção condicionada

Define-se o limiar de "partícula massuda" por

```
N * P(m > m_big) = 2      =>      p := P(m > m_big) = 2/N ,      m_big = F^-1( 1 - 2/N )
```

Para `N = 1000`, `alpha = 2.35`, `R = 1000` **[T]**:

| grandeza | valor |
|---|---|
| `p` | `2.0e-3` |
| `m_big` | `2.7509063196e10 kg` = `27.509 * PARTICLE_MASS` |
| `m_big / m_min` | `96.658` |
| `<m \| m > m_big>` | `6.1912383238e10 kg` = `61.912 * PARTICLE_MASS` |
| `<m \| m <= m_big>` | `8.7793109572e8 kg` = `0.87793 * PARTICLE_MASS` |
| fração da massa total na cauda | `0.123825` |

Construção proposta, a ser verificada:

1. sortear `k` de `Binom(N, p)` **renormalizada em `{1,2,3}`**;
2. sortear `k` massas da cauda `[m_big, m_max]` por CDF inversa no subintervalo;
3. sortear `N - k` massas do corpo `[m_min, m_big]` por CDF inversa no subintervalo.

### 2.6 Prova de que a construção é exatamente a condicional

**Afirmação a verificar:** a construção acima produz exatamente a amostra da lei de potência
condicionada a `k ∈ {1,2,3}`, e não uma aproximação.

**Veredito: a afirmação é CORRETA, sob uma condição que a construção enunciada não menciona e que
este documento torna normativa** (passo 4 da Seção 2.7).

**Prova.** Sejam `m_1, ..., m_N` i.i.d. de densidade `p`. Sejam `A = (m_big, m_max]`,
`B = [m_min, m_big]`, `p = P(A)`, e `K = #{i : m_i ∈ A}`. Então `K ~ Binom(N, p)` exatamente, por
definição de `K` como soma de `N` indicadores i.i.d. de Bernoulli(`p`).

Fixe `k` e um subconjunto `S ⊂ {1..N}` com `|S| = k`. A densidade conjunta condicionada ao evento
`{K = k}` é

```
  prod_i p(x_i) * 1{ #(x_i in A) = k }  /  P(K = k)
```

e, restrita ao setor em que o conjunto de índices na cauda é exatamente `S`, fatoriza como

```
  [ prod_{i in S}   p(x_i) 1_A(x_i) ] * [ prod_{i not in S} p(x_i) 1_B(x_i) ]
  --------------------------------------------------------------------------
                    C(N,k) * p^k * (1-p)^(N-k)

= (1/C(N,k)) * prod_{i in S} [ p(x_i) 1_A(x_i) / p ] * prod_{i not in S} [ p(x_i) 1_B(x_i) / (1-p) ]
```

Lê-se diretamente: **(i)** o conjunto `S` é uniforme entre os `C(N,k)` subconjuntos de tamanho `k`;
**(ii)** dado `S`, as `k` massas da cauda são i.i.d. de `p` truncada a `A`; **(iii)** as `N-k`
restantes são i.i.d. de `p` truncada a `B`; **(iv)** os três blocos são independentes. **[T]**

Condicionar adicionalmente a `K ∈ {1,2,3}` não altera nada dentro de cada `k`: pela regra da
cadeia, `P(· | K ∈ {1,2,3}, K = k) = P(· | K = k)`, e a lei marginal de `K` passa a ser

```
P(K = k | K in {1,2,3}) = C(N,k) p^k (1-p)^(N-k) / sum_{j=1}^{3} C(N,j) p^j (1-p)^(N-j)
```

que é precisamente a binomial renormalizada do passo 1. **[T]** Logo a construção é exata. **Não é
uma aproximação.**

**A condição que faltava.** A prova entrega `S` **uniforme**. A construção enunciada não diz onde as
`k` massas da cauda são gravadas. Se forem gravadas nos slots `0..k-1`, o **multiconjunto** de
massas continua exato, mas a lei conjunta `(posição, massa)` **não** é a condicional: com a semente
de posições fixa em `SEED = 20190222`, as partículas massudas cairiam sempre nas mesmas posições
`r_0, r_1, r_2`, em toda realização do ensemble. Isso não é um detalhe estético — invalida qualquer
estudo de ensemble sobre o efeito da posição inicial dos corpos massudos. A permutação uniforme dos
slots é **normativa** (Seção 2.7, passo 4) e testável (`INV-13`).

### 2.7 Algoritmo normativo de amostragem de massas

Determinístico e reprodutível, exatamente nesta ordem de sorteios, em fp64, com um `Generator`
dedicado (**não** o mesmo fluxo usado pelas posições — ver a nota ao final):

```
rng_m = numpy.random.default_rng(mass_seed)          # PCG64, fluxo separado

# 1. k, da binomial renormalizada em {1,2,3}
w_k   = [ C(N,k) * p^k * (1-p)^(N-k)  for k in (1,2,3) ]
w_k  /= sum(w_k)
k     = 1 + searchsorted( cumsum(w_k), rng_m.random() )     # 1 sorteio

# 2. k massas da cauda [m_big, m_max]
u_tail = rng_m.random(k)                                     # k sorteios
m_tail = inv_cdf_on( u_tail, m_big, m_max )

# 3. N-k massas do corpo [m_min, m_big]
u_body = rng_m.random(N - k)                                 # N-k sorteios
m_body = inv_cdf_on( u_body, m_min, m_big )

# 4. colocacao uniforme nos slots  <-- OBRIGATORIO (Secao 2.6)
masses          = concatenate( [ m_tail, m_body ] )
perm            = rng_m.permutation(N)                       # 1 bloco
masses          = masses[perm]
```

Pontos normativos:

- **O passo 4 não é opcional.** Sem ele a construção não é a condicional da Seção 2.6.
- **O fluxo de massas é separado do fluxo de posições.** Motivo: `cold_sphere` consome exatamente
  `3N` sorteios do fluxo de `SEED` (Seção 5.2 de `integradores.md`) e o arquivo
  `data/ic_sphere_N1000_seed20190222.npz` é verificado bit a bit contra esse fluxo. Injetar sorteios
  de massa no mesmo `Generator` quebraria essa verificação. `mass_seed` é um parâmetro
  independente, com valor padrão declarado na Seção 8.
- `k` é sorteado **antes** das massas, e `m_tail` **antes** de `m_body`. A ordem é normativa: ela
  fixa o fluxo, e o fluxo é o que torna a condição inicial reprodutível.
- A amostragem é feita **sempre em fp64**, mesmo quando a integração roda em fp32, pelo mesmo motivo
  da Seção 5.2 de `integradores.md`.

**Distribuição de `k`** (`N = 1000`, `p = 2e-3`) **[T]**:

| `k` | `Binom(N, p)` | renormalizada em `{1,2,3}` |
|---|---|---|
| 0 | `0.13506452` | — |
| 1 | `0.27067039` | `0.37476530` |
| 2 | `0.27094160` | `0.37514082` |
| 3 | `0.18062773` | `0.25009388` |
| >= 4 | `0.14269576` | — |

`P(K ∈ {1,2,3}) = 0.72223972`. **O condicionamento descarta 27.8% da medida natural.**

### 2.8 O que o condicionamento custa — dito com número

Duas consequências, ambas obrigatórias de registrar em qualquer figura ou tabela de massas:

**(a) A amostra realizada NÃO é a lei de Salpeter.** É Salpeter condicionada a um evento de
probabilidade `0.722`. Em particular `E[K | K ∈ {1,2,3}] = 1.875329`, contra `E[K] = N p = 2`
sem condicionamento. **[T]**

**(b) A massa média realizada não é `PARTICLE_MASS`.** O déficit de corpos de cauda arrasta a média:

```
E[M_tot | K in {1,2,3}] - N*<m>  =  ( E[K|.] - Np ) * ( <m|A> - <m|B> )  =  -7.609e9 kg
```

isto é **`-0.7609%`** da massa total nominal. **[T]** `E[M_tot | ·] = 9.9239075e11 kg`.

**Decisão normativa.** `m_min` é resolvido pela lei **não condicionada** (Seção 2.4), porque essa é a
lei-mãe bem definida, e o viés de `-0.76%` é **reportado**, não corrigido. Corrigir `m_min` para
compensar o condicionamento tornaria `m_min` dependente de `N` e de `p` por uma via não fechada, e
faria a "lei de Salpeter" do documento deixar de ser Salpeter. **[T]**

Consequência direta: **`t_ff` é calculado da massa total REALIZADA, nunca da nominal** (Seção 2.10).
Esse é o mecanismo pelo qual o viés de `-0.76%` entra corretamente em todos os observáveis
normalizados, em vez de ficar escondido.

### 2.9 Variância da massa total — a intuição do enunciado está pela metade

**Afirmação a verificar:** "para `alpha = 2.35` a integral de massa é dominada pelo extremo inferior,
então a soma deve concentrar bem".

**Veredito: a premissa é verdadeira, a conclusão NÃO se segue. REFUTADA.**

A premissa está certa: a massa por intervalo logarítmico é `m p(m) ∝ m^(1-alpha) = m^(-1.35)`,
estritamente decrescente, e de fato só `12.38%` da massa está acima de `m_big`. **[T]**

Mas a concentração da soma é governada pela **variância**, cujo integrando é
`m² p(m) ∝ m^(2-alpha) = m^(-0.35)`, com primitiva `∝ m^(0.65)` — **dominada pelo extremo
SUPERIOR**. Os dois expoentes caem em lados opostos do ponto de virada, e é o segundo que decide.
**[T]**

Números **[T]**:

```
<m>            = 1.0000e9 kg
<m^2>          = 1.4826e19 kg^2
Var(m)         = 1.3826e19 kg^2
sd(m)          = 3.7184e9 kg
sd(m) / <m>    = 3.7184        <-- o desvio-padrao de UMA massa e 3.7x a propria media
```

Para `N = 1000` i.i.d. **sem** condicionamento, `CV(M_tot) = sd(m)/(<m> sqrt(N)) = 11.76%`. **[T]**

Com o condicionamento em `k ∈ {1,2,3}` (que remove a cauda `K >= 4`, a maior fonte de dispersão):

```
E[M_tot | .]   = 9.9239075e11 kg
sd(M_tot | .)  = 9.1699e10 kg
CV(M_tot | .)  = 9.2402%
```

decomposta em: sorteios do corpo `27.90%`, sorteios da cauda `45.10%`, contagem `K` `26.99%`. **[T]**

**Leitura obrigatória.** Uma dispersão de `9.24%` na massa total entre realizações **não** é
"concentrar bem". Ela é a razão pela qual `t_ff` tem de vir da massa realizada, e é o piso da
dispersão de ensemble de qualquer observável normalizado por `M_tot`. Reportar `t_collapse/t_ff`
com quatro casas decimais sobre uma única semente, como faz `INV-9`, deixa de ser legítimo quando as
massas são sorteadas: a variação entre sementes domina a precisão numérica por muitas ordens de
grandeza.

**Onde a intuição falha, em uma frase:** o extremo inferior domina *quanta massa existe*; o extremo
superior domina *quanto ela varia*. As duas perguntas têm respostas opostas com `alpha = 2.35`, e a
soma obedece à segunda.

### 2.10 `t_ff` da massa realizada

`t_ff` deixa de ser uma constante do projeto e passa a ser uma função da realização:

```
M_real   = sum_i m_i                      (massa efetivamente sorteada)
rho_real = M_real / ( (4/3) * pi * R_0^3 )
t_ff     = sqrt( 3*pi / (32 * G * rho_real) )
```

Como `t_ff ∝ M_real^(-1/2)`, a dispersão relativa de `t_ff` é **metade** da de `M_real`:

```
sd(t_ff)/t_ff = (1/2) * CV(M_tot|.) = 4.62%                       [T]
```

Pontos normativos:

- `R_0 = SPHERE_RADIUS = 6.2035049090 m` permanece fixo. É a esfera que é reutilizada, não a
  densidade. `rho` passa a flutuar com a massa sorteada.
- **Todo eixo temporal adimensional usa o `t_ff` da própria realização.** Usar `T_FF = 2.1007035`
  (o valor de massa nominal) numa execução com massas sorteadas introduz um erro sistemático de até
  `~5%` no eixo, que é da mesma ordem do efeito de `Q` que a Seção 3 pretende medir. Este é o erro
  mais fácil de cometer e o mais difícil de ver num gráfico.
- `V_CHAR`, `L_SCALE` e `U_MIN_BOUND` são igualmente derivados de `M_real` quando usados como
  normalizações de teste. `U_MIN_BOUND` passa a ser
  `-G * (sum_{i<j} m_i m_j) / eps = -G * (M_real^2 - sum_i m_i^2) / (2 eps)`, que **não** é
  `-G m^2 N(N-1)/(2 eps)` — ver a emenda a `INV-10` na Seção 9.

---

## 3. Velocidades iniciais

### 3.1 Distribuição alvo, e a distinção que este documento existe para fixar

Maxwelliana isotrópica **truncada**. A densidade no **espaço de velocidades** é

```
f(v) d^3v  proporcional a  exp( -|v|^2 / (2 sigma^2) ) d^3v ,     |v| <= v_cut
f(v) = 0 ,                                                        |v| >  v_cut
```

com `sigma` a dispersão por componente e `v_cut = f_cut * v_esc` o teto de truncamento.

**Ponto normativo, e o mais fácil de enunciar errado em toda esta especificação.**

O requisito do projeto é *"quanto mais rápida a partícula, menor a chance de aparecer"*. Isso é:

- **VERDADEIRO** para a densidade no espaço de velocidades `f(v) ∝ exp(-|v|²/2σ²)`, que é
  **estritamente decrescente na rapidez** `|v|`, com máximo em `v = 0`. **[T]**
- **FALSO** para a distribuição da **rapidez** `p(|v|) ∝ |v|² exp(-|v|²/2σ²)`, que se anula em
  `|v| = 0`, **cresce** até `|v| = sqrt(2) sigma` e só então decresce. **[T]**

As duas afirmações são simultaneamente corretas porque medem coisas diferentes: `f` é densidade por
unidade de **volume** `d³v`, `p` é densidade por unidade de **rapidez** `d|v|`, e
`p(s) = 4 pi s² f(s)` — o fator `4 pi s²` é a área da casca esférica, e é ele, não a exponencial,
que domina em `s` pequeno.

Enunciados **permitidos** no relatório:

- "a densidade no espaço de velocidades é estritamente decrescente na rapidez";
- "para dois elementos de volume `d³v` de mesmo tamanho, o mais afastado da origem é o menos
  provável";
- "a rapidez mais provável é `sqrt(2) sigma`, e rapidezes acima e abaixo dela são menos frequentes".

Enunciados **proibidos**:

- "partículas rápidas são mais raras que partículas lentas" (falso abaixo de `sqrt(2) sigma`);
- "a maioria das partículas está quase parada" (falso: `p(s) -> 0` quando `s -> 0`);
- qualquer frase que não diga **em relação a que medida** a monotonicidade vale.

Valores para o `Q` padrão (Seção 3.4): `sigma = 0.7604389 m/s`, rapidez mais provável
`sqrt(2) sigma = 1.0754 m/s`, `v_rms = 1.267482 m/s`. **[T]**

### 3.2 Uma distribuição estritamente decrescente na rapidez seria melhor? Não

Pergunta legítima: se o requisito é "mais rápida, menos provável", por que não escolher uma `p(|v|)`
estritamente decrescente e acabar com a ambiguidade?

**Veredito: rejeitado, e a razão é física, não estética.**

Uma `p(s)` estritamente decrescente em três dimensões exige `f(v) = p(s)/(4 pi s²)`, isto é, uma
densidade no espaço de velocidades que **diverge como `s^-2`** na origem. Consequências:

1. `f` é integrável (`s^-2 * s² = const`), então a distribuição existe — mas é **singular**: há uma
   cúspide de densidade infinita em `v = 0`. Nenhum sistema físico em equilíbrio, ou próximo dele,
   tem isso.
2. Não é a distribuição de equilíbrio de hamiltoniano algum. A Maxwelliana é; é por isso que ela
   aparece. Trocá-la por uma forma escolhida para satisfazer uma frase em português abandonaria a
   única justificativa física da escolha.
3. A relaxação violenta que o próprio colapso produz destruiria a cúspide em menos de um tempo de
   cruzamento, de modo que o esforço não sobreviveria ao primeiro `t_ff`. **[A]** — a medição que
   decide é o histograma de `|v|` em `t = 0` contra `t = 1 t_ff` no estágio 1.

**Decisão: mantém-se a Maxwelliana truncada.** O requisito do projeto é satisfeito exatamente, na
medida `d³v`, e a Seção 3.1 fixa a linguagem para que ninguém afirme a versão errada depois. O
truncamento, aliás, satisfaz o requisito **mais** fortemente que a gaussiana pura: acima de `v_cut`
a probabilidade é exatamente zero, não apenas pequena.

### 3.3 Ordem das operações — e o que fica exato

Construção proposta: amostrar gaussiana por componente, rejeitar `|v_i| > f_cut * v_esc`, subtrair a
velocidade média ponderada por massa, reescalar globalmente para `K = Q|U|/2`.

**Veredito sobre a ordem: a ordem proposta está CORRETA e é normativa.** A verificação, passo a
passo, e o que cada passo preserva ou destrói:

**Passo 1 — amostragem e rejeição.** Sortear `v_i` gaussiano isotrópico e **redesenhar o vetor
inteiro** enquanto `|v_i| > v_cut`.

> **Normativo:** a rejeição é por **rapidez da partícula**, com resorteio das três componentes
> juntas. Truncar componente a componente produz um cubo em vez de uma bola no espaço de
> velocidades, e destrói a isotropia. Este é um erro que passa despercebido em qualquer teste que
> olhe só `|v|` marginal.

**Passo 2 — subtração da média ponderada por massa.** Com `V = (sum_i m_i v_i) / M_tot`,
faz-se `v_i <- v_i - V`. Isto zera `P` exatamente em aritmética exata, e reduz `K` por um valor
exato:

```
sum_i m_i |v_i - V|^2 = sum_i m_i |v_i|^2 - 2 V . (sum_i m_i v_i) + M_tot |V|^2
                      = sum_i m_i |v_i|^2 - M_tot |V|^2                              [T]
```

isto é, `K` cai exatamente `(1/2) M_tot |V|²`. A perda relativa é `~1/N = 0.1%`. **[T]** Não é
preciso compensá-la: o passo 3 a absorve.

**Passo 3 — reescalonamento global.** Com `K'` a energia cinética após o passo 2 e
`K_alvo = Q|U|/2`, faz-se `v_i <- lambda v_i` com `lambda = sqrt(K_alvo / K')`.

> **Por que esta ordem e não a inversa.** O reescalonamento é linear e homogêneo, logo
> `sum_i m_i (lambda v_i) = lambda sum_i m_i v_i = 0`: **ele preserva `P = 0` exatamente**. **[T]**
> A subtração da média, em contrapartida, **muda** `K`. Portanto subtrair-e-depois-reescalar deixa
> as duas condições exatas simultaneamente; reescalar-e-depois-subtrair deixa `K` errado por
> `(1/2)M_tot|V|²`, isto é, por `~0.1%`. A ordem importa e é normativa.

**Passo 4 — nada.** Não há passo 4. Em particular, **não se repete a rejeição após o
reescalonamento** (ver 3.3.1).

**Resumo do que fica exato e do que fica aproximado:**

| grandeza | estado após o passo 3 | marcação |
|---|---|---|
| `P = sum_i m_i v_i` | **exato** (zero, a menos de arredondamento) | **[T]** |
| `K = Q\|U\|/2` | **exato** (por construção do `lambda`) | **[T]** |
| `Q = 2K/\|U\|` realizado | **exato** | **[T]** |
| isotropia | exata em distribuição; a realização tem ruído `O(N^-1/2)` | **[T]** |
| teto `\|v_i\| <= v_cut` | **aproximado**: vale para a amostra pré-escala, e passa a `lambda*v_cut` | **[T]** |
| forma maxwelliana truncada | exata a menos do fator de escala global `lambda` | **[T]** |

#### 3.3.1 O teto é aproximado, e por quanto

O reescalonamento move o teto: após o passo 3 a cota efetiva é `lambda * v_cut`, não `v_cut`.
Iterar (rejeitar de novo, subtrair de novo, reescalar de novo) **não** é aceito: o ponto fixo dessa
iteração não é a maxwelliana truncada de parâmetro nenhum, e a distribuição resultante deixa de ter
forma fechada — perde-se justamente a propriedade que justifica a escolha.

**Decisão normativa: o teto é imposto exatamente sobre a amostra pré-escala e a implementação
reporta `lambda` e `max_i |v_i| / v_esc`.** A magnitude do desvio é calculável e pequena:

```
Var(v^2)/<v^2>^2 = 2/3        para uma maxwelliana
sd relativa da media amostral de v^2  =  sqrt( 2 / (3N) )  =  2.582%     (N = 1000)
sd relativa de lambda = metade disso  =  1.291%                                     [T]
```

Logo `lambda = 1 +- 1.3%` (1 sigma), **desde que `sigma` seja resolvido corretamente** (Seção 3.4).
Reportar "as velocidades estão limitadas a `0.5 v_esc`" sem essa ressalva é uma afirmação falsa em
cerca de um terço das realizações. A forma permitida é: "o teto `0.5 v_esc` é imposto na amostragem;
o reescalonamento de virial o desloca em `+-1.3%` (1 sigma), medido e reportado por execução".

#### 3.3.2 `sigma` sai de uma equação implícita, não de `<v²> = 3 sigma²`

Se `sigma` for escolhido pela relação **não truncada** `<v²> = 3 sigma²`, o `lambda` do passo 3 não é
uma correção de ruído amostral: é uma correção **sistemática**, porque o truncamento reduz `<v²>`.
Para o `Q` padrão o fator seria `sqrt(3/2.778) = 1.039`, isto é, `3.9%` — três vezes o ruído
amostral, e sistemático em vez de aleatório.

**Normativo:** `sigma` é resolvido pela equação implícita do segundo momento truncado,

```
sigma^2 * h( v_cut / sigma )  =  <v^2>_alvo  =  Q |U(r^0)| / M_real

              int_0^x  t^4 exp(-t^2/2) dt
h(x)  =  ---------------------------------          h(inf) = 3
              int_0^x  t^2 exp(-t^2/2) dt
```

`h` é estritamente crescente, e `sigma -> sigma² h(v_cut/sigma)` é estritamente crescente em `sigma`,
logo a raiz é única e obtida por bisseção. **[T]**

Valores de `h` **[T]**:

| `x` | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 | 8.0 |
|---|---|---|---|---|---|---|
| `h(x)` | `1.170398` | `1.830313` | `2.391337` | `2.753459` | `2.982850` | `3.000000` |

`|U(r^0)|` é o potencial da **configuração inicial com as massas realizadas**, calculado com
`eps = SOFTENING`, e `M_real` é a massa realizada. Nenhum dos dois é uma constante do projeto quando
o espectro de massas está ligado.

### 3.4 O teto de truncamento impõe um teto em `Q` — e a combinação proposta o viola

**Veredito: `f_cut = 0.5` com `Q = 0.5` é INADMISSÍVEL. Vetado.**

O truncamento impõe um supremo em `<v²>` que nenhum `sigma` alcança. Quando `sigma -> infinito` com
`v_cut` fixo, a exponencial fica plana sobre `[0, v_cut]` e a distribuição degenera na **bola
uniforme** no espaço de velocidades, `p(s) ∝ s²`, cujo segundo momento é

```
<v^2>_sup = (3/5) v_cut^2 = (3/5) f_cut^2 v_esc^2                                  [T]
```

Traduzindo em `Q`, com `v_esc = sqrt(2 G M_real / R_0)`:

```
Q_sup(f_cut) = (3/5) f_cut^2 v_esc^2 M_real / |U_0|  =  2.009056 * f_cut^2         [T]
```

(o coeficiente `2.009056` usa `|U_0| = 6.4260397026e12 J` e `M_real = 1e12 kg`, isto é, o caso de
massas iguais; com massas sorteadas ele flutua com `M_real/|U_0|`.)

| `f_cut` | `Q_sup` (limite degenerado) | `Q` usável (`x_c >= 2`) | `Q` usável (`x_c >= 3`) |
|---|---|---|---|
| `0.5` | `0.502264` | `0.3830` | `0.2561` |
| `0.7` | `0.984438` | `0.7508` | `0.5020` |
| `1.0` | `2.009056` | `1.5322` | `1.0244` |

**Portanto `Q = 0.5` com `f_cut = 0.5` está exatamente sobre a singularidade `Q_sup = 0.502264`.**
A bisseção de 3.3.2 ou não converge, ou converge para um `sigma` enorme cuja distribuição não é
maxwelliana coisa alguma: é a bola uniforme, com `f(v)` **plana**, o que aniquila a única
propriedade — monotonicidade estrita de `f` na rapidez — que a Seção 3.1 acabou de estabelecer como
o conteúdo do requisito do projeto. Medido na bisseção: `sigma` corre até o limite superior da busca
e a fração rejeitada vai a `99.4%`. **[T]**

**Critério de não degenerescência (normativo).** Exige-se `x_c := v_cut / sigma >= 2`, isto é,
truncamento não mais apertado que `2 sigma`. Abaixo disso a fração rejeitada passa de `26%` e a
forma da distribuição é dominada pelo corte, não pela exponencial. Em termos fechados:

```
Q  <=  1.532168 * f_cut^2            (condicao x_c >= 2)                           [T]
```

A implementação **deve verificar essa desigualdade e levantar `ValueError`** quando violada, nomeando
`Q`, `f_cut` e o `Q` máximo admissível. Falhar a bisseção silenciosamente, ou devolver a bola
uniforme, é inaceitável (`INV-16`).

### 3.5 Valores fixados

**`Q_DEFAULT = 0.25`, `F_CUT_DEFAULT = 0.5`.**

Justificativa, em ordem de peso:

1. **Preserva `f_cut = 0.5` do enunciado** e cai confortavelmente dentro do regime não degenerado:
   `x_c = 3.0500`, com `2.55%` de rejeição. A distribuição é reconhecivelmente maxwelliana.
2. **Está longe dos dois extremos degenerados.** `Q = 0` reproduz exatamente o colapso frio já
   medido e não acrescenta física nova; `Q = 1` é o equilíbrio de virial, em que não há colapso e o
   observável-chefe (`t_collapse`) deixa de existir.
3. **O sistema permanece fortemente ligado:** `E = K + U = (Q/2 - 1)|U| = -0.875 |U| < 0`. Não há
   evaporação apreciável no horizonte de `3 t_ff`.
4. **O efeito é mensurável mas não destrutivo.** As predições de 3.6 dão desvios de `+15%` em
   `t_collapse` e `+34%` em `r_half,min` — muito acima do ruído numérico (`4` casas decimais em
   `t_collapse`, `INV-9`) e muito abaixo de descaracterizar o colapso.
5. **`dt = 5.0e-4 s` permanece válido.** O critério de `dt` de `integradores.md` §4.4 é a frequência
   máxima do tensor de maré, que depende de `eps` e da densidade local, não da velocidade inicial.
   O critério adicional que velocidades introduzem é `|v| dt << eps`: com `v_rms = 1.27 m/s`,
   `|v| dt = 6.3e-4 m = 0.013 eps`. Folga de duas ordens. **[T]**

Valores derivados, massas iguais (`M_real = 1e12`, `|U_0| = 6.4260397026e12 J`) **[T]**:

| grandeza | valor |
|---|---|
| `v_esc = sqrt(2 G M / R_0)` | `4.6386556804 m/s` = `1.414229 V_CHAR` |
| `v_cut = 0.5 v_esc` | `2.319328 m/s` |
| `K_alvo = Q\|U_0\|/2` | `8.032550e11 J` |
| `v_rms` | `1.267482 m/s` |
| `sigma` (raiz da eq. implícita) | `0.7604389 m/s` |
| `x_c = v_cut/sigma` | `3.0500` |
| `h(x_c)` | `2.77814` |
| fração rejeitada | `2.5529%` |
| rapidez mais provável `sqrt(2) sigma` | `1.0754 m/s` |

Alternativa permitida para estudo de sensibilidade, **não** para os testes: `Q = 0.5` com
`f_cut = 0.7` (`sigma = 1.0794174`, `x_c = 3.0082`, rejeição `2.86%`). **[T]**

### 3.6 `Q = 0` deve reproduzir o colapso frio exatamente

**Normativo:** quando `Q == 0.0`, o amostrador retorna `zeros((N,3))` **sem consumir sorteio algum**
e sem executar os passos 1 a 3. Justificativas:

- `lambda = sqrt(0/K') = 0` daria `0.0 * v_i`, que em IEEE-754 produz `-0.0` sempre que `v_i < 0`.
  `-0.0 == 0.0` compara verdadeiro, mas os bits diferem, e o projeto verifica condições iniciais
  **bit a bit** contra `data/ic_sphere_N1000_seed20190222.npz` (Seção 5.3 de `integradores.md`).
- Consumir sorteios deslocaria o fluxo do `Generator` e quebraria a reprodutibilidade de qualquer
  sorteio posterior no mesmo fluxo.
- A divisão por `K'` é indefinida se a amostra degenerar; o curto-circuito remove a possibilidade.

Isto é testável de forma dura (`INV-17`): com `Q = 0` e massas iguais, o estado devolvido deve ser
**bit a bit idêntico** ao de `cold_sphere`.

### 3.7 Predições falsificáveis sobre `Q`

Duas predições baratas, ambas derivadas, ambas verificáveis com uma execução por valor de `Q` no
estágio 1. Ambas são **[T]** quanto à derivação e **[A]** quanto à validade da hipótese de
homologia que carregam; a medição que as decide está dita em cada uma.

#### (P1) Instante do cruzamento pelo centro: `t_collapse ∝ (1 - Q)^(-1/2)`

Da identidade de Lagrange–Jacobi para o momento de inércia `I = sum_i m_i |r_i|²`:

```
(1/2) d^2 I / dt^2 = 2K + U                                                        [T]
```

Em `t = 0`, com `2K_0 = Q|U_0|` e `U_0 = -|U_0|`:

```
(d^2 I / dt^2)|_0 = 2 (Q - 1) |U_0|                                                [T]
```

isto é, a curvatura inicial do colapso é reduzida pelo fator `(1-Q)` em relação ao caso frio. Além
disso `(dI/dt)|_0 = 2 sum_i m_i r_i . v_i = 0` em média, por isotropia (o resíduo amostral é
`O(N^-1/2)`). Se o colapso for aproximadamente auto-semelhante sob essa reescala — a hipótese
**[A]** —, o tempo escala como o inverso da raiz da "gravidade efetiva":

```
t_collapse(Q) / t_ff  ~  1.0361 / sqrt(1 - Q)
```

usando `t_collapse(0)/t_ff = 1.0361` de `INV-9`. Predições:

| `Q` | `t_collapse/t_ff` previsto |
|---|---|
| `0.00` | `1.0361` (medido, `INV-9`) |
| `0.10` | `1.0921` |
| `0.25` | **`1.1964`** |
| `0.50` | `1.4653` |
| `0.75` | `2.0722` |

**Como falsificar.** Rodar o estágio 1 em `Q ∈ {0, 0.1, 0.25}` com a mesma semente de posições e
massas iguais, e ajustar `t_collapse(Q) sqrt(1-Q)/t_ff`. A predição é que essa combinação é
**constante e igual a `1.036`**. Banda de aceitação proposta: constância dentro de `+-15%` sobre
`Q ∈ [0, 0.5]`. **[A]** A divergência em `Q -> 1` é uma propriedade correta da fórmula (em `Q = 1` o
sistema está virializado e não colapsa), e serve como teste de sanidade qualitativo.

#### (P2) Raio de meia-massa mínimo: barreira de momento angular linear em `Q`

Em `Q = 0` o piso de `r_half,min = 0.3472 m` é fixado por discreteza e softening. Com dispersão
isotrópica, cada partícula ganha momento angular específico `j = r v_t`, com
`<v_t²> = (2/3)<v²> = (2/3) Q (3/5) G M / R_0`. O pericentro de uma órbita quase radial com esse `j`
num campo interno de massa `M` é `r_p ≈ j²/(2GM)`, o que dá, para `r ~ R_0`:

```
r_p  ~  Q R_0 / 5                                                                  [T]
```

Somando em quadratura com o piso de `Q = 0` (as duas causas são independentes, **[A]**):

| `Q` | `Q R_0 / 5` | `r_half,min` previsto |
|---|---|---|
| `0.00` | `0.0000` | `0.3472` (medido) |
| `0.10` | `0.1241` | `0.3687` |
| `0.25` | `0.3102` | **`0.4656`** |
| `0.50` | `0.6204` | `0.7109` |

**Como falsificar.** A predição de conteúdo é: `r_half,min` **cresce monotonicamente com `Q`**, é
aproximadamente **linear em `Q`** acima de `Q ≈ 0.3` (onde a barreira domina o piso), e o coeficiente
angular é da ordem de `R_0/5 = 1.24 m` por unidade de `Q`. Banda proposta para `Q = 0.25`:
`r_half,min ∈ [0.40, 0.55] m`. **[A]** Se `r_half,min` **cair** com `Q`, a construção de velocidades
está errada — o candidato mais provável é rejeição por componente (Seção 3.3, passo 1), que produz
velocidades preferencialmente diagonais em vez de isotrópicas.

**Aviso.** Com o espectro de massas ligado, a dispersão de ensemble em `t_ff` (`4.62%`, Seção 2.9)
e em `r_half,min` domina esses efeitos para `Q <~ 0.1`. As duas predições devem ser medidas com
**massas iguais** para isolar o efeito de `Q`, e só depois repetidas com o espectro ligado.

---

## 4. Colisões

### 4.1 Raio de contato — a grandeza, não o número

```
R_i = R_ref * ( m_i / m_bar )^(1/3) ,      m_bar = PARTICLE_MASS = 1e9 kg
R_ref = chi * eps                          eps = SOFTENING = 5.0e-2 m
```

O expoente `1/3` corresponde a densidade material constante. `chi` é **adimensional** e é o único
parâmetro livre desta subseção. Este documento **não fixa `chi`**; fixa o que medi-lo significa.

**Grandeza a medir (normativa):** o **número médio de colisões por partícula durante o primeiro
rebote**,

```
N_coll_per_particle := 2 * (numero de eventos detectados em [t_bounce - 0.05 t_ff, t_bounce + 0.05 t_ff])
                       / N_live
```

com `t_bounce = t_collapse`. **Faixa-alvo: `N_coll_per_particle ∈ [0.5, 2]`.** Abaixo de `0.5` as
colisões são invisíveis e a extensão é decoração; acima de `2` a colisão, e não a gravidade, passa a
governar o núcleo (Seção 4.2).

**Estimativa a priori** — núcleo em compressão máxima, `500` partículas dentro de
`r_half,min = 0.3472 m`, massas iguais, `v_rel = sqrt(2) V_CHAR = 4.6386 m/s`, janela de rebote
`0.21 s = 0.1 t_ff` **[A]**:

| `chi` | `R (m)` | `R_i+R_j (m)` | `sigma_geo (m²)` | `Gamma (s^-1)` | livre-caminho `(s)` | `N_coll_per_particle` |
|---|---|---|---|---|---|---|
| `1` | `0.0500` | `0.1000` | `3.142e-2` | `415.6` | `2.41e-3` | **`87.3`** |
| `0.5` | `0.0250` | `0.0500` | `7.854e-3` | `103.9` | `9.63e-3` | `21.8` |
| `0.25` | `0.0125` | `0.0250` | `1.964e-3` | `25.98` | `3.85e-2` | `5.46` |
| `0.1` | `0.0050` | `0.0100` | `3.142e-4` | `4.156` | `0.241` | **`0.873`** |
| `0.05` | `0.0025` | `0.0050` | `7.854e-5` | `1.039` | `0.963` | `0.218` |

com `n_v = 500 / ((4/3) pi r_half,min³) = 2852 m^-3` e `Gamma = n_v sigma_geo v_rel`. **[T]** dado
`n_v`, **[A]** quanto a `n_v` e à janela.

O valor que dá exatamente `N_coll_per_particle = 1` é **`chi = 0.107`**, isto é `R_ref ≈ eps/9.3`.
**[T]**

**Valor de partida recomendado: `chi = 0.1`, `R_ref = 5.0e-3 m = eps/10`.** Marcado **[A]**; o
estágio 2 o confirma ou o corrige, medindo `N_coll_per_particle` diretamente.

**Discrepância registrada.** `integradores.md` §4.3 declara densidade numérica de núcleo
`~1.4e3 m^-3` e separação local `~0.089 m`; o cálculo direto acima com `500` partículas dentro de
`r_half,min` dá `2852 m^-3` e `0.0705 m`. Fator `2` em densidade. Este documento usa `2852`; se a
medição do estágio 2 confirmar `1.4e3`, todas as taxas desta tabela caem por `2` e `chi` recomendado
sobe para `≈ 0.15`. **[A]** — a medição que decide é o histograma de densidade local no instante
`t_collapse`.

**A densidade nominal do projeto NÃO fornece um raio físico.** Uma esfera de `1e9 kg` à densidade
`rho = 1e9 kg/m³` teria raio `0.6203504909 m = 12.407 eps`, e duas dessas se tocariam à separação
interpartícula média de `1.0 m`. **[T]** Isto significa que `rho = 1e9 kg/m³` é a densidade **do
sistema**, um artefato de contabilidade geométrica herdado da grade de 2019, e não uma densidade
material. Não existe raio de contato "físico" a ser derivado dela: `chi` é irredutivelmente um
parâmetro de modelo, e deve constar de todo artefato de saída, como `eps`.

**Dispersão de raios induzida pelo espectro de massas** — `(m/m_bar)^(1/3)` **[T]**:

| corpo | `m/m_bar` | `(m/m_bar)^(1/3)` | `R` com `chi=0.1` |
|---|---|---|---|
| `m_min` | `0.2846` | `0.6578` | `3.289e-3 m` |
| `m_bar` | `1.0000` | `1.0000` | `5.000e-3 m` |
| `m_big` | `27.509` | `3.0187` | `1.509e-2 m` |
| `m_max` | `284.60` | `6.5778` | `3.289e-2 m` |

Com `chi = 1` o corpo mais massudo teria `R = 0.3289 m`, praticamente igual a `r_half,min = 0.3472`:
**a "partícula" teria o tamanho do núcleo colapsado**. Mais um argumento contra `chi` grande.

### 4.2 A tensão com o softening — veredito

O glossário registra que abaixo de `d <~ eps` o sistema não é o problema newtoniano de massas
pontuais. Segue o argumento: se `R_i + R_j > eps`, nenhum par atinge separação `< eps` sem colidir
antes, e o softening deixa de atuar. **Isso é um argumento a favor de contato acima de `eps`?**

**Veredito: NÃO. Vetado.** O argumento é formalmente correto e a conclusão é errada, porque o preço
é maior que o defeito que ele compra.

O que o argumento acerta: com `chi >= 1` a região regularizada torna-se inacessível, e o sistema
passa a ser gravidade newtoniana **não suavizada** mais contato de esfera rígida — semanticamente
mais limpo. E a cota de aceleração de par não piora: em `d = 2 eps` a aceleração suavizada é
`4.775 m/s²` contra a cota de Plummer `10.27 m/s²`, então o critério de `dt` de §4.4 de
`integradores.md` sobrevive. **[T]**

O que o argumento ignora: **o custo em regime dinâmico é catastrófico.** Pela tabela de 4.1, `chi=1`
dá `87` colisões por partícula por rebote — o núcleo sofre `~2e4` eventos em `0.2 s`. Com qualquer
probabilidade de fusão não desprezível, as `500` partículas do núcleo coalescem em punhado de corpos
dentro de um único rebote. O objeto simulado deixa de ser um colapso frio de N corpos e passa a ser
um problema de coalescência colisional. **Trocar uma imprecisão de `d <~ eps` — que afeta a força
de par num volume desprezível do espaço de fases — por uma mudança de regime do sistema inteiro é
uma troca ruim, e é uma troca que o argumento não declara estar fazendo.**

Não existe `chi` que evite os dois chifres. Este documento escolhe um e o declara:

| regime | `chi` | o que acontece | estatuto |
|---|---|---|---|
| **perturbativo** | `~0.1` | `~1` colisão/partícula/rebote; a colisão é um canal raro sobre o colapso existente; o resultado colisionless permanece o limite `chi -> 0` | **padrão** |
| intermediário | `0.25 – 0.5` | `5 – 22` colisões/partícula/rebote; o núcleo é colisional; o colapso ainda é reconhecível | sensibilidade |
| dominado | `>= 1` | `>= 87`; coalescência; o softening nunca atua | estudo separado |

**Consequência que o regime perturbativo obriga a declarar.** Com `chi = 0.1`, `R_i + R_j = 0.01 m`
é **cinco vezes menor** que `eps = 0.05 m`. Portanto **as colisões acontecem dentro da região
regularizada**: a velocidade de impacto e o potencial de par no contato são os do potencial de
Plummer, não os de massas pontuais. Isso não é um erro — é a semântica correta do modelo — mas
**tem de estar na prosa**, porque muda os números: a profundidade do poço de par no contato é
`G m_i m_j / sqrt((R_i+R_j)² + eps²)` e não `G m_i m_j / (R_i+R_j)`, uma diferença de fator `5.1`
para o par de massa média. **[T]** Ver a Seção 4.6, onde essa distinção entra explicitamente no
parâmetro de regime, e a emenda a `integradores.md` §10 na Seção 9.

### 4.3 Detecção varrida — a álgebra está correta

Sobre um passo `[0, h]`, com `dr = r_j - r_i` e `dv = v_j - v_i` constantes (movimento retilíneo), a
separação ao quadrado é `|dr + t dv|²`, uma parábola **convexa** em `t`, com mínimo em

```
t* = clamp( - (dr . dv) / |dv|^2 ,  0 ,  h )
```

**Verificação: a fórmula está correta.** Derivando, `d/dt |dr + t dv|² = 2 (dr + t dv) . dv = 0` dá
`t = -(dr.dv)/|dv|²`; convexidade garante que o clamp a `[0,h]` devolve o mínimo **sobre o
intervalo**, não um extremo espúrio. **[T]** Verificado numericamente contra minimização por grade
de `2e6` pontos em 5 configurações aleatórias: coincidência exata em `t*` e no valor do mínimo, em
todos os casos, inclusive nos dois em que o clamp foi ativado. **[M]**

Colide se `|dr + t* dv| < R_i + R_j`.

**Casos degenerados, normativos:**

- `|dv|² == 0`: adotar `t* = 0`. Sem esta guarda há divisão por zero. Ocorre exatamente quando as
  duas partículas têm velocidade idêntica bit a bit — raro, mas produzido de propósito por qualquer
  teste com condição inicial simétrica.
- **Guarda de aproximação:** o par só é candidato se `dr . dv < 0` no **início** do passo (isto é,
  aproximando-se). Sem essa guarda, um par que acabou de colidir e ainda está sobreposto dispara de
  novo no passo seguinte, produzindo colisões "pegajosas" — o artefato mais comum de detectores de
  contato ingênuos.

**A hipótese de movimento retilíneo é exata para o *drift* do Verlet.** No esquema KDK a atualização
de posição é `r^(n+1) = r^n + h v^(n+1/2)` com `v^(n+1/2)` constante ao longo do subpasso. Se `dv`
for tomado como a diferença das velocidades de **meio passo**, o varrido não é uma aproximação:
descreve exatamente a trajetória que o integrador percorre. **[T]** É por isso que a Seção 4.5
insere a colisão **dentro** do *drift*, e não depois do passo completo.

O resíduo de curvatura, se alguém preferir aplicar o teste ao passo inteiro, é
`(1/2)|a_rel| h² = 3.1e-6 m` com `a_rel = 25 m/s²` e `h = 5e-4 s` — `3e-4` do diâmetro de contato em
`chi = 0.1`. Desprezível. **[T]**

### 4.4 Tunelamento — possível, e é isso que fixa `dt`

Deslocamento relativo por passo com `dt = 5.0e-4 s` **[T]**:

| `\|u\|` | deslocamento/passo |
|---|---|
| `4.64 m/s` (típico no núcleo) | `2.32e-3 m` |
| `10 m/s` | `5.0e-3 m` |
| `30 m/s` (extremo estimado no rebote) | `1.5e-2 m` |

Definindo o número de Courant colisional `C_coll = |u| dt / (R_i + R_j)` **[T]**:

| `chi` | `R_i+R_j` | `C_coll` a `4.64 m/s` | `C_coll` a `30 m/s` |
|---|---|---|---|
| `1` | `0.100` | `0.023` | `0.150` |
| `0.5` | `0.050` | `0.046` | `0.300` |
| `0.25` | `0.025` | `0.093` | `0.600` |
| `0.1` | `0.010` | `0.232` | **`1.500`** |

**Resposta: sim, o tunelamento é possível no regime perturbativo.** Com `chi = 0.1` e o par mais
rápido, `C_coll = 1.5 > 1`: o par atravessa a zona de contato inteira dentro de um passo. O teste
varrido **detecta** o mínimo (é essa a razão de ele existir), mas ao fim do passo os corpos já se
separaram, a normal de fim de passo aponta para o lado errado, e a guarda de aproximação descarta o
evento. A colisão é perdida silenciosamente.

Para o par **mais leve** (`R_i+R_j = 2 * 0.1 * eps * 0.6578 = 6.58e-3 m`), `C_coll = 1.14` mesmo com
`dt = 2.5e-4`.

**Decisão normativa: `DT_COLLISION = 1.25e-4 s`**, isto é, `DT_COLLAPSE / 4`. Derivação:

```
exige-se  C_coll < 1  para o par mais desfavoravel:
    dt < (R_i+R_j)_min / |u|_max  =  6.58e-3 / 30  =  2.19e-4 s
escolhe-se  dt = 1.25e-4 s   ->   C_coll,max = 30 * 1.25e-4 / 6.58e-3 = 0.57      [T]
```

`1.25e-4` é escolhido em vez de `2.0e-4` por dois motivos operacionais: divide `OUT_DT = 1e-2 s`
exatamente (`80` passos por saída) e mantém a relação de potência de dois com `DT_COLLAPSE`, o que
preserva a comparabilidade com as escadas de `integradores.md` §8.2. Custo: `4x` o número de passos,
`50400` passos para `3 t_ff`.

`|u|_max = 30 m/s` é **[A]**, estimado por `2 sqrt(2 G M_core / r_half,min)` com
`M_core = 5e11 kg`. A medição que o decide é o máximo de `|v_i - v_j|` sobre pares candidatos ao
longo do estágio 2. **`C_coll` medido é um invariante obrigatório (`INV-18`)**: se exceder `1` em
qualquer passo, `dt` está grande demais e os resultados colisionais do estágio são inválidos.

### 4.5 Onde a colisão entra no passo, e pareamento disjunto

**Esquema normativo.** A colisão é inserida **dentro do *drift*** do Verlet, não após o passo:

```
v^(n+1/2)            = v^n + (h/2) a^n
(r^(n+1), v^(n+1/2)) = C_h( r^n, v^(n+1/2), m )        # passe de colisao (abaixo)
a^(n+1)              = A(r^(n+1), m)
v^(n+1)              = v^(n+1/2) + (h/2) a^(n+1)
```

O passe `C_h` faz, nesta ordem:

1. **Detecção.** Para todo par, `t*` e teste de contato da Seção 4.3, com `dv = v^(n+1/2)`.
2. **Ordenação e aceitação gulosa.** Ordenar os candidatos pela chave lexicográfica `(t*, i, j)`.
   Percorrer em ordem, aceitando o par se **nenhum** dos dois já foi reivindicado neste passe.
3. **Resolução.** Para cada par aceito: avançar **os dois participantes** de `t*` (movimento
   retilíneo), aplicar o mapa de desfecho ali, e avançar os corpos resultantes o restante `h - t*`.
4. **Drift simples.** Todos os não participantes recebem `r <- r + h v`.

Por que **dentro** do drift e não depois do passo:

- O impulso é aplicado **na configuração de contato**, com a normal `n = (dr + t* dv)/|dr + t* dv|`
  paralela à separação real no ponto de aplicação. **É essa paralelidade, e só ela, que faz o
  choque elástico conservar `L` exatamente** (Seção 4.8). Aplicar o impulso no fim do passo com a
  normal de contato destrói a conservação de `L`; aplicar com a normal de fim de passo a preserva
  mas perde a colisão sempre que houver tunelamento (4.4).
- Nenhuma sobreposição residual é produzida: os corpos são separados pela própria dinâmica após
  `t*`. Isso torna desnecessário — e **proibido** — qualquer reposicionamento *ad hoc*.
- A estrutura KDK permanece intacta. Com zero eventos, `C_h` é a identidade composta com o drift, e
  o passo é **bit a bit** o passo de `integradores.md` §3.3 (`INV-30`).

> **Veto sobre "separar até contato exato".** A proposta de deslocar os corpos sobrepostos até se
> tocarem é rejeitada. Deslocar posições altera `U`: afastar corpos torna `U` menos negativo, isto é,
> **injeta energia mecânica** sem física compensatória. No esquema acima a sobreposição nunca se
> forma, então não há o que corrigir. Se por alguma razão futura o esquema mudar e a sobreposição
> reaparecer, a correção obrigatória **não** é reposicionar: é aplicar a guarda de aproximação e
> deixar a dinâmica separar os corpos.

**O pareamento guloso conserva massa.** Cada par aceito envolve dois corpos ainda não reivindicados;
os mapas de desfecho atuam sobre conjuntos **disjuntos** de slots, logo comutam, e a massa total é a
soma sobre grupos disjuntos. Cada um dos três desfechos conserva massa individualmente (Seção 4.8).
Logo o passe conserva. **[T]** Em ponto flutuante o resíduo é de um arredondamento por evento
(Seção 7, `TOL-MASS-SUM`).

**O pareamento guloso é uma aproximação, e isso é declarado.** Depois que `(i,j)` colide em `t*_1`,
a trajetória de `i` muda, de modo que um candidato posterior `(i,k)` com `t*_2 > t*_1` deveria ser
**reavaliado**, não simplesmente rejeitado. Rejeitá-lo adia o evento para o passo seguinte. Isso é
legítimo enquanto a multiplicidade por passo for baixa, e a validade é **medida**, não presumida:

```
f_reject := (candidatos rejeitados por disjuncao) / (candidatos detectados)
```

`f_reject` é reportado por execução; **`INV-19`** exige `f_reject <= 0.05` sobre toda a execução.
Acima disso, `dt` é grande demais para o modelo colisional, exatamente como `C_coll > 1`.

**Determinismo.** A chave de ordenação inclui `(i, j)` precisamente para desempatar `t*` idênticos
em ponto flutuante. Sem isso, a ordem de aceitação depende da estabilidade do algoritmo de ordenação
e o resultado deixa de ser reprodutível entre dispositivos. Normativo.

### 4.6 Parâmetro de regime `x`

O desfecho não é sorteado uniformemente: é enviesado por um parâmetro adimensional que compara a
energia do impacto com a energia que mantém o par unido.

**Proposta original:** `E_coh = (1/2) mu v_coh²`, `x = |u|²/v_coh²`, pura coesão.
**Veredito: aceita com uma emenda obrigatória — o termo gravitacional NÃO pode ser omitido.**

Definição normativa:

```
E_bind = (1/2) mu v_coh^2   +   G m_i m_j / sqrt( (R_i+R_j)^2 + eps^2 )
T_cm   = (1/2) mu |u|^2 ,        mu = m_i m_j / (m_i + m_j) ,   u = v_j - v_i

x := T_cm / E_bind
```

**A massa reduzida cancela.** Substituindo `mu = m_i m_j / M` com `M = m_i + m_j`:

```
x = |u|^2 / ( v_coh^2 + v_esc_eff^2 ) ,     v_esc_eff^2 := 2 G M / sqrt( (R_i+R_j)^2 + eps^2 )   [T]
```

Um único escalar `v_crit = sqrt(v_coh² + v_esc_eff²)` e `x = (|u|/v_crit)²`. O custo computacional
do termo gravitacional é **uma soma e uma raiz**, sobre quantidades que a detecção já calculou.

**Por que o termo gravitacional é obrigatório — três razões, em ordem de peso.**

**(1) Sem ele, `x >= 1` identicamente, e o canal de fusão fica inalcançável.** Para um par que chega
ao contato vindo de separação grande, a conservação de energia do par dá
`T_cm(contato) = T_inf + G m_i m_j / sqrt((R_i+R_j)² + eps²)`. Se `E_bind` contiver **apenas** esse
termo gravitacional, então `x = 1 + T_inf/E_grav >= 1` para todo par não ligado. **[T]** Um mapa
centrado em `x = 1` teria toda a faixa visitada de um lado só — exatamente a "decoração" que se quer
evitar. É o termo de coesão que abaixa o piso:

```
x_floor = v_esc_eff^2 / ( v_coh^2 + v_esc_eff^2 )  <  1                            [T]
```

**(2) Ele não é desprezível numericamente.** `v_esc_eff` para `chi = 0.25` **[T]**:

| par | `chi=1` | `chi=0.25` | `chi=0.1` |
|---|---|---|---|
| `m_min`–`m_min` | `0.959` | `1.202` | `1.227` |
| `m_bar`–`m_bar` | `1.545` | `2.185` | `2.288` |
| `m_max`–`m_bar` | `9.988` | `18.866` | `24.651` |
| `m_max`–`m_max` | `10.732` | `21.025` | `30.324` |

Contra `v_coh = V_CHAR = 3.28 m/s`, o termo gravitacional vale `44%` de `v_crit²` para o par de massa
média e é **41 vezes maior** que a coesão para o par mais massudo. Omiti-lo não é uma simplificação;
é apagar o termo dominante da população que mais importa.

**Nota estrutural:** `v_esc_eff` é limitado por `sqrt(2GM/eps)` quando `R_i+R_j << eps` — é o
**softening**, não o raio de contato, que fixa a profundidade do poço no regime perturbativo. Por
isso `v_esc_eff` varia pouco com `chi` para corpos leves (`1.20 -> 1.23` de `chi=0.25` a `0.1`) e
muito para corpos massudos. Consequência direta da Seção 4.2.

**(3) Ele é o mecanismo que regula o crescimento descontrolado** — ver 4.11.

**Adimensionalização de `v_coh`.** `v_coh = COH_VELOCITY_FACTOR * V_CHAR`, com
`V_CHAR = sqrt(G M_real / R_0)` recalculado da massa realizada (Seção 2.10).
**`COH_VELOCITY_FACTOR = 1.0` por padrão.** **[A]** Justificativa: `V_CHAR` é a única escala de
velocidade que o problema já tem; com esse valor os três canais recebem cada um `>= 5%` dos eventos
ao longo do colapso (Seção 4.7). A medição que o confirma é a estatística de canais do estágio 3
(`INV-26`).

### 4.7 O mapa `x -> (p_el, p_fus, p_frag)`

Construção normativa: **softmax sobre três escores lineares em `s = ln x`**, com meia-largura `b` do
platô elástico e largura `w` de suavização.

```
s = clamp( ln(x) , -30 w , +30 w )

S_fus  = -s            S_el = b            S_frag = +s

p_c = exp(S_c / w) / sum_c' exp(S_c'/ w)
```

equivalentemente `(p_fus, p_el, p_frag) ∝ ( x^(-1/w),  e^(b/w),  x^(+1/w) )`.

Propriedades, todas verificadas numericamente sobre `x ∈ [1e-6, 1e6]` em `601` pontos **[M]**, e
demonstráveis **[T]**:

- **Soma exatamente 1** por construção do softmax.
- **Estritamente positivas para todo `x`**: a exponencial nunca se anula. Nenhum canal é jamais
  proibido.
- **Suave em `ln x`**: `p_c` é analítica em `s`, e `s = ln x` é a variável natural porque `x` varre
  décadas.
- **Monotonicidade:** `p_fus` estritamente decrescente em `x`, `p_frag` estritamente crescente,
  `p_el` máxima em `x = 1` (onde `Z = 2 cosh(s/w) + e^(b/w)` é mínima). Este é todo o conteúdo
  físico do mapa; o resto é calibração.
- **`w -> infinito` recupera o caso uniforme** `(1/3, 1/3, 1/3)` exatamente, com `b` fixo. Isto é
  o **caso de controle**, e `w = inf` é entrada válida.
- **Travessias:** fusão iguala elástica em `x = e^(-b)`, fragmentação iguala elástica em `x = e^(+b)`.

**Estabilidade numérica (normativa):** avaliar por *log-sum-exp* (subtrair `max_c S_c/w` antes de
exponenciar). O `clamp` em `+-30 w` existe para honrar literalmente o requisito "nenhuma
probabilidade exatamente zero": sem ele, `x -> 0` faz `p_frag` subnormalizar a `0.0` em ponto
flutuante. Com o clamp, a menor probabilidade representável é `e^(-60) = 8.76e-27` **[M]**,
confortavelmente acima do menor normal de fp32 (`1.18e-38`).

**Valores padrão: `b = ln 3 = 1.0986123`, `w = 3.0`.** **[A]** — a calibração que os fixa é
`INV-26`. Com esses valores **[M]**:

| `x` | `p_fus` | `p_el` | `p_frag` |
|---|---|---|---|
| `0.01` | `0.7368` | `0.2290` | `0.0342` |
| `0.1` | `0.5305` | `0.3552` | `0.1143` |
| `0.33` | `0.4042` | `0.4028` | `0.1930` |
| `1` | `0.2905` | `0.4190` | `0.2905` |
| `3` | `0.1938` | `0.4031` | `0.4031` |
| `6.4` | `0.1404` | `0.3758` | `0.4838` |
| `20` | `0.0814` | `0.3187` | `0.5999` |
| `100` | `0.0342` | `0.2290` | `0.7368` |

Para comparação, `w = 1` produz um mapa muito mais duro (`p_fus = 0.0164` em `x = 6.4`) e `w = 5`
um mapa quase uniforme (`p_fus = 0.2038` em `x = 6.4`). `w = 3` é o compromisso: viés de fator
`~14` em `p_fus` entre `x = 1` e `x = 226`, com nenhum canal abaixo de `5%` na faixa visitada.

**Sorteio (normativo).** Um único uniforme por evento aceito, de um `Generator` dedicado, consumido
na ordem da lista ordenada por `(t*, i, j)` da Seção 4.5. Desfecho: fusão se `u < p_fus`, elástica se
`u < p_fus + p_el`, senão fragmentação. Determinístico dada a semente.

### 4.8 Faixa de `x` que a simulação realmente visita

Esta é a pergunta que decide se o mapa é física ou enfeite. Com `chi = 0.25`,
`v_coh = V_CHAR = 3.28 m/s`, e `x = (|u_inf|² + v_esc_eff²)/(v_coh² + v_esc_eff²)` **[T]**:

| par | `v_esc_eff` | `x_floor` | `\|u_inf\|=0.2` | `1` | `4.64` | `10` | `30` |
|---|---|---|---|---|---|---|---|
| `m_min`–`m_min` | `1.201` | `0.118` | `0.122` | `0.200` | `1.88` | `8.31` | `73.9` |
| `m_bar`–`m_bar` | `2.185` | `0.307` | `0.310` | `0.372` | `1.69` | `6.74` | `58.2` |
| `m_max`–`m_max` | `21.03` | `0.976` | `0.976` | `0.978` | `1.02` | `1.20` | `2.96` |

**Conclusão: a faixa visitada é `x ∈ [0.12, ~75]`, quase três décadas, cavalgando `x = 1`.** Nenhum
canal fica faminto:

- encontros lentos de corpos leves (`x ≈ 0.12–0.4`): `p_fus ≈ 0.39–0.53` — fusão dominante;
- encontros típicos no núcleo (`x ≈ 1.7`): `p ≈ (0.242, 0.414, 0.344)` — os três ativos;
- encontros rápidos (`x ≈ 20–75`): `p_frag ≈ 0.60–0.70` — fragmentação dominante.

**Sem o termo de coesão, `x_floor = 1` e a primeira linha desapareceria inteira.** É esse cálculo,
e não uma preferência, que justifica a emenda da Seção 4.6.

**Ressalva obrigatória.** Corpos massudos têm `x` **preso perto de `1`** (`0.976` a `2.96` em toda a
faixa de `|u_inf|`), porque `v_esc_eff` domina numerador e denominador simultaneamente. Isto tem duas
leituras, ambas verdadeiras: (i) o mapa perde poder discriminante para a população massuda; (ii) essa
população recebe `p_fus ≈ p_frag ≈ 0.29`, o que a auto-regula — ver 4.11.

**Estatuto epistêmico do mapa, dito sem rodeios.** O mapa é uma **interpolação fenomenológica**, não
uma teoria de colisões. Seu conteúdo físico é a **ordenação** (fusão em `x` baixo, fragmentação em
`x` alto, elástica no meio) e a monotonicidade; os valores de `b`, `w` e `v_coh` são calibração.
Consequência normativa para o relatório: **as frações de canal realizadas são uma saída da
parametrização, jamais uma predição física.** É permitido escrever "sob o modelo fixado na Seção 4.7,
`X%` dos eventos foram fusões"; é proibido escrever "o colapso produz `X%` de fusões".

### 4.9 Os três desfechos

Notação comum: `M = m_i + m_j`, `mu = m_i m_j / M`, `u = v_j - v_i`, `V = (m_i v_i + m_j v_j)/M`,
`T_cm = (1/2) mu |u|²`. Tudo avaliado na configuração de contato, em `t*`.

#### (1) Elástica

```
n     = ( dr + t* dv ) / | dr + t* dv |            (normal de contato, unitaria)
J     = 2 mu ( u . n ) n                           (impulso)
v_i' = v_i + J / m_i
v_j' = v_j - J / m_j
```

Equivalentemente `v_i' = v_i + (2 m_j / M)(u.n) n`, `v_j' = v_j - (2 m_i / M)(u.n) n`, e a velocidade
relativa vira `u' = u - 2(u.n) n`: componente normal invertida, tangencial preservada.

Conservações, **todas exatas em aritmética exata** **[T]**, verificadas numericamente **[M]**:

| grandeza | resíduo relativo medido |
|---|---|
| massa | `0` (exato) |
| `P` | `5.06e-17` |
| `L` | `3.63e-17` |
| `K` | `0` (exato) |
| `sum_i m_i r_i` | `0` (exato, posições inalteradas) |
| `U` | inalterada (posições inalteradas no instante `t*`) |

`K` é conservada porque `|u'| = |u|` (reflexão), e `K = (1/2)M|V|² + (1/2)mu|u|²` com ambos os termos
preservados — **para qualquer `n` unitária**, o que torna a conservação robusta a erro na normal.
`L` é conservada porque `ΔL = -(dr) x J` e `J` é paralelo a `dr` **no ponto de aplicação**; é
exatamente por isso que a Seção 4.5 aplica o impulso em `t*` e não no fim do passo. **[T]**

**Consequência: a colisão elástica é um mapa que conserva `E`, `P` e `L` exatamente.** `E_int` não é
alterada. Nenhum tratamento de sobreposição residual é necessário nem permitido (Seção 4.5).

#### (2) Fusão

```
m_novo = M
r_novo = ( m_i r_i + m_j r_j ) / M          (centro de massa do par, em t*)
v_novo = V = ( m_i v_i + m_j v_j ) / M
slot liberado: m = 0                         (Secao 5)
```

**Verificação da afirmação do enunciado: "energia cinética não pode ser conservada aqui, e a perda é
exatamente `(1/2) mu |v_i - v_j|²`".**

**Veredito: a primeira metade está CORRETA; a segunda está INCOMPLETA.**

Correto: `K = (1/2)M|V|² + (1/2)mu|u|²` antes, `(1/2)M|V|²` depois, logo

```
Delta K = - (1/2) mu |u|^2 = - T_cm            exatamente                          [T]
```

Verificado numericamente: `ΔK = -1.418940e9 J` contra `-(1/2)mu|u|² = -1.418940e9 J`, concordância
relativa `1.01e-15`. **[M]** Conservar `K` exigiria `u = 0`, isto é, que os corpos já viajassem
juntos — não é uma fusão, é uma redefinição de rótulo.

Incompleto: a variação de `U` **não** é apenas "o termo de potencial mútuo que desaparece". Ela tem
duas partes, e a segunda não se anula:

```
Delta U =  + G m_i m_j / sqrt( d_ij^2 + eps^2 )                                  (termo mutuo, some)
           - sum_{k != i,j} G m_k [ M / sqrt(d_ck^2+eps^2)
                                    - m_i/sqrt(d_ik^2+eps^2) - m_j/sqrt(d_jk^2+eps^2) ]   (terceiros)
```

Os termos de terceiro corpo são nulos apenas no limite `d_ij -> 0`; para `d_ij = R_i+R_j` finito eles
existem, e **não têm forma fechada em função de `mu` e `|u|`**. **[T]**

**Isto vindica a solução do acumulador, mas por uma razão diferente da alegada:** não é que a perda
seja "`T_cm` mais um termo conhecido"; é que **não existe forma fechada**, e a única contabilidade
correta é numérica. Ver 4.10.

**Conservações da fusão** **[T]**, verificadas **[M]**:

| grandeza | estatuto | resíduo medido |
|---|---|---|
| massa | exata | `0` |
| `P` | exata | `0` |
| `sum_i m_i r_i` (centro de massa) | **exata** | `0` |
| `K` | **destruída** por `-T_cm` | concordância `1.01e-15` |
| `L` | **destruída** por `-mu (dr x u)` | ver abaixo |

**A fusão destrói momento angular, e o enunciado do projeto não menciona isso.** Escrevendo
`r_a = r_c + delta_a`, `v_a = V + w_a`:

```
Delta L = - mu ( dr x u )                                                          [T]
```

isto é, exatamente o momento angular **interno** (de spin) do par em torno do próprio centro de
massa. Fisicamente ele vai para a rotação do corpo fundido, que o modelo não representa. Verificado
numericamente: `|ΔL|` previsto `9.901784e6` contra medido `9.901784e6`, e `|ΔL|/|L|` do par igual a
`6.03e-3` — **não** desprezível. **[M]**

Cota por evento: `|ΔL| <= mu (R_i+R_j) |u|`. Para `mu = 5e8`, `R_i+R_j = 0.01`, `|u| = 5`:
`2.5e7 kg m²/s`, isto é `1.2e-6` de `L_SCALE`. Somando incoerentemente sobre `1e4` eventos:
`~1.2e-4` de `L_SCALE` — acima de qualquer tolerância de `INV-3`. **Tem de ser contabilizado**
(Seção 4.10).

**Colocação no centro de massa é obrigatória.** É a única posição que preserva `sum_i m_i r_i`,
portanto a única que não desloca o centro de massa do sistema. Verificado: resíduo `0`. **[M]**
Bônus algébrico: a fusão **comuta com o drift**, pois
`M(r_c + (h-t*)V) = m_i(r_i + (h-t*)v_i) + m_j(r_j + (h-t*)v_j)` identicamente. **[T]** Logo a
ordem entre "fundir" e "terminar o drift" é irrelevante — uma propriedade que os testes podem usar.

#### (3) Fragmentação

```
f      ~ U(0.1, 0.9)                       (razao de massa, sorteada)
m_a    = f * M
m_b    = M - m_a                            <-- NAO (1-f)*M ; ver nota de ponto flutuante
mu'    = m_a m_b / M
u'     = |u| * sqrt(mu/mu') * n_iso          n_iso isotropica, sorteada
v_a    = V + (m_b/M) u'
v_b    = V - (m_a/M) u'
r_a    = r_c + (m_b/M) (R_a+R_b) * u'/|u'|
r_b    = r_c - (m_a/M) (R_a+R_b) * u'/|u'|
```

**Verificação da construção proposta: massa, momento e `T_cm` conservados exatamente.**

**Veredito: CONFIRMADO, os três.** **[T]**, verificado **[M]**:

| grandeza | demonstração | resíduo medido |
|---|---|---|
| massa | `m_a + m_b = M` por construção | `0` |
| `P` | `m_a v_a + m_b v_b = M V + (m_a m_b/M - m_b m_a/M) u' = P` | `5.06e-17` |
| `T_cm` | `(1/2)mu'\|u'\|² = (1/2)mu'\|u\|²(mu/mu') = (1/2)mu\|u\|²` | `0` (exato) |
| `K` total | `K = (1/2)M\|V\|² + T_cm`, ambos preservados | `3.43e-16` |
| `sum_i m_i r_i` | colocação simétrica com `m_a delta_a + m_b delta_b = 0` | `2.19e-17` |

Verificado também `|u'|/|u| = sqrt(mu/mu')` com concordância a 6 casas. **[M]**

**Nota de ponto flutuante (normativa):** calcular `m_b = M - m_a`, **não** `(1-f)*M`. A primeira
forma introduz um único arredondamento e mantém `m_a + m_b = M` a menos de `1` ulp; a segunda
introduz dois e o resíduo dobra.

**Colocação dos fragmentos e o que ela faz com `U`.** Os fragmentos são postos **em contato**
(`|r_a - r_b| = R_a + R_b`, com os novos raios) e **ao longo de `u'`**, portanto separando-se. Isso:
(i) preserva `sum_i m_i r_i` exatamente; (ii) impede recolisão imediata, reforçando a guarda de
aproximação; (iii) altera `U` — o termo mútuo passa de `-G m_i m_j/sqrt(d_ij²+eps²)` para
`-G m_a m_b/sqrt((R_a+R_b)²+eps²)`, e os termos de terceiro corpo mudam. Como na fusão, **sem forma
fechada** — vai para `E_int`.

**`L` também é destruída, pelo mesmo termo.** Como `r_a - r_b` é paralelo a `u'`, o momento angular
interno **depois** é `mu' (r_a - r_b) x u' = 0`, logo `ΔL = -mu (dr x u)`, idêntico ao da fusão.
Medido: `|ΔL|/|L| = 6.03e-3`. **[M]** Mesmo tratamento (Seção 4.10).

**Ressalva física, obrigatória de declarar.** Conservar `T_cm` exatamente significa que esta
"fragmentação" **não dissipa nada**: é um evento de redistribuição de massa com energética elástica.
Uma fragmentação real é disruptiva e dissipa. O modelo, como especificado, é o que o enunciado pediu
e está algebricamente correto, mas **não é fragmentação no sentido astrofísico** e não pode ser
descrito como tal. É esta escolha, e não outra, que permite a `E_int` decrescer (Seção 4.10).
Extensão opcional, **desligada por padrão**: impor `|u'| = |u| sqrt(mu/mu') sqrt(1-eta)` com
`eta ∈ [0,1)` a fração dissipada, que então vai para `E_int`. O diagnóstico que decide se `eta > 0`
é necessário está em `INV-23`.

**Sem piso de massa.** A fragmentação repetida pode levar massas abaixo de `m_min` indefinidamente.
O processo **se auto-extingue** (secção de choque `∝ R² ∝ m^(2/3)`), mas não há piso declarado. A
implementação reporta `min_i m_i` ao longo do tempo e o número de corpos abaixo de `m_min`
(`INV-31`). Se o moinho for excessivo, a mitigação é um termo de supressão contínuo
`S_frag <- S_frag - Lambda(M)`, que preserva a positividade estrita — **não** um corte duro, que a
violaria.

### 4.10 Os acumuladores `E_int` e `L_spin`

**A proposta do enunciado está certa na forma e ERRADA no ponto de avaliação. Emenda obrigatória.**

O enunciado propõe `E_int += E_mec(antes) - E_mec(depois)` "avaliado sobre o passe inteiro". Se
"passe inteiro" significar o passo de integração, a contabilidade absorve também o **erro de
truncamento do integrador**, e `E_total = K + U + E_int` passa a ser conservada trivialmente,
medindo exatamente nada. O diagnóstico-chefe do projeto seria destruído pela própria máquina
construída para salvá-lo.

**Definição normativa: os acumuladores são avaliados ATRAVÉS DO MAPA DE DESFECHO, com as posições
congeladas em `t*`, e nunca ao longo do passo.**

```
para cada evento aceito, na configuracao de contato em t*:

    Delta K   = K(depois)  - K(antes)          (so os participantes contribuem)
    Delta U   = U(depois)  - U(antes)          (termo mutuo + termos de terceiro corpo)

    E_int    += - ( Delta K + Delta U )
    L_spin   += - ( L_orb(depois) - L_orb(antes) )  =  + mu ( dr(t*) x u )   para fusao e fragmentacao
                                                       0                     para elastica
```

`ΔU` é computável em `O(N)` por evento, porque só os termos que envolvem `i` e `j` mudam:

```
U_antes(i,j) = - G m_i m_j / sqrt(d_ij^2+eps^2)
               - sum_{k != i,j} G m_k [ m_i/sqrt(d_ik^2+eps^2) + m_j/sqrt(d_jk^2+eps^2) ]

U_depois(i,j) = - G m_a m_b / sqrt(d_ab^2+eps^2)                    (zero na fusao)
                - sum_{k != i,j} G m_k [ m_a/sqrt(d_ak^2+eps^2) + m_b/sqrt(d_bk^2+eps^2) ]
```

O custo total do passe é `O(n_events * N)`, não `O(N²)`, e não entra em `n_force`.

**Quantidades conservadas por construção:**

```
E_total = K + U + E_int          conservada exatamente pelo mapa de colisao        [T]
L_total = L_orb + L_spin         conservada exatamente pelo mapa de colisao        [T]
```

**Aviso normativo sobre o que essas identidades testam.** Por construção, elas são **tautologias no
evento**: não testam a física do desfecho, testam apenas que o acumulador foi somado. O que testa a
física são os invariantes analíticos **por desfecho** de `INV-20`, `INV-21`, `INV-22`
(`ΔK = -T_cm` na fusão, `ΔT_cm = 0` na fragmentação, `ΔK = ΔL = 0` na elástica). Os dois conjuntos
são complementares e **ambos** obrigatórios; nenhum substitui o outro.

**`E_int` é um acumulador com sinal, e o sinal é um diagnóstico.**

- **Fusão sempre incrementa `E_int` por um valor não negativo** (ignorando terceiros): o incremento é
  `T_cm - E_grav`, e a conservação de energia do par que cai desde separação grande garante
  `T_cm >= E_grav` no contato. **[T]** Portanto fusões só dissipam.
- **Fragmentação decrementa `E_int`**, porque `ΔK = 0` e `ΔU > 0` para `f != 0.5` (o produto
  `m_a m_b = f(1-f)M² <= M²/4` cai, logo `|U_mutuo|` cai). **[T]**
- Nada garante `E_int(t) >= 0`. Se `E_int` ficar negativa, **o modelo de colisão está injetando
  energia mecânica no sistema a partir de um reservatório que nunca foi carregado.**

**Critério normativo (`INV-23`): `min_t E_int(t) >= -delta |E_0|` com `delta = 1e-3`.** Abaixo disso,
a dinâmica está sendo movida pelo modelo de colisão e não pela gravidade, e os resultados do estágio
são inválidos. A mitigação declarada é `eta > 0` na fragmentação (Seção 4.9).

`E_int(t)/|E_0|` é, além de guarda, um **resultado físico por direito próprio**: é o orçamento de
dissipação do colapso colisional, e deve ir para o relatório como curva.

**Ordem de grandeza por evento.** Uma fusão de dois corpos de `1e9 kg` a `|u| = 5 m/s` dissipa
`T_cm = 6.25e9 J`, contra `|E_0| = 6.43e12 J`: **`0.1%` da energia total do sistema em um único
evento.** **[T]** Com milhares de eventos, `E_int` torna-se comparável a `E_0`. Colisões **não** são
uma perturbação no orçamento de energia; são o termo dominante.

### 4.11 O que sobrevive do simpletismo, e qual diagnóstico substitui o antigo

**Enunciado preciso do que se perde.**

`velocity_verlet` é simplético e conserva exatamente um hamiltoniano sombra `H_h = H + O(h²)`; é
disso, e não de nenhuma propriedade de `E`, que decorre a banda limitada de `|ΔE/E₀|`. O mapa de
colisão `C_h` não é o fluxo-`h` de hamiltoniano suave algum. A análise de erro para trás não
atravessa o evento, e `H_h` é redefinido a cada evento.

| afirmação | com colisões elásticas apenas | com fusão/fragmentação |
|---|---|---|
| `velocity_verlet` é simplético no trecho entre eventos | **sobrevive** **[T]** | **sobrevive** **[T]** |
| `P` conservado a menos de arredondamento | **sobrevive** **[T]** | **sobrevive** **[T]** |
| `L` conservado exatamente | **sobrevive** **[T]** | **NÃO sobrevive**; `L_orb` salta `-mu(dr x u)` |
| `E` conservada pelo mapa de evento | **sobrevive, exatamente** **[T]** | **NÃO sobrevive**; `E_mec` salta |
| banda limitada de `\|ΔE/E₀\|` sem deriva secular | **plausível** **[A]** | **NÃO sobrevive** |
| valores `[M]` de `longrun_energy.csv` | não comparáveis (trajetória diferente) | não comparáveis |

Detalhando as duas linhas que importam:

**Elástica apenas.** O choque elástico de esferas rígidas é o limite de um potencial repulsivo
íngreme e o mapa preserva a forma simplética; ele conserva `E`, `P` e `L` exatamente (Seção 4.9).
Logo o **único** erro de energia continua sendo o truncamento de Verlet entre eventos, e o
comportamento limitado é o esperado — **[A]**, porque o argumento do hamiltoniano sombra não se
aplica literalmente a um mapa não suave. A medição que decide é o estágio 2: `|ΔE/E₀|` ao longo de
`10 t_ff` com detecção ligada e todos os desfechos elásticos, comparado com a mesma execução sem
colisões. Se a banda permanecer limitada, o **[A]** vira **[M]**; se aparecer deriva secular, o
mapa elástico está implementado errado (candidato mais provável: normal avaliada fora de `t*`).

**Fusão/fragmentação.** `E_mec = K + U` exibe uma **escada**: cada evento é um degrau finito, de até
`~1e-3 |E_0|`. Não há banda limitada, não há oscilação em torno de valor fixo, e a razão
"pico/final ≈ 60" que distingue simpléticos de não simpléticos em `INV-4` **deixa de existir como
teste**. Qualquer critério de `INV-4` aplicado a `E_mec` numa execução com fusão falha contra uma
implementação correta.

**Diagnóstico substituto — quatro peças, todas obrigatórias:**

- **(D1) `|Delta E_total / E_total(0)|` com `E_total = K + U + E_int`.** É o sucessor direto da
  banda antiga: conservado exatamente pelos eventos, portanto sua deriva mede **só** o integrador
  entre eventos. Os critérios **qualitativos** de `INV-4` (`velocity_verlet` não monótono, final
  muito menor que o pico; `euler` monótono crescente; `rk4` derivando negativo) transferem-se para
  `E_total`. Os valores **[M]** de `integradores.md` **não** se transferem: a trajetória é outra.
- **(D2) `E_int(t)/|E_0|`.** O orçamento de dissipação. Resultado físico, não só guarda.
- **(D3) Resíduo por evento.** Recalcular `K + U + E_int` imediatamente antes e depois de cada mapa,
  com posições congeladas, e exigir concordância ao nível da precisão de máquina (`TOL-EVENT-CONS`).
  É isto que detecta um acumulador somado com sinal trocado ou um termo de terceiro corpo esquecido.
- **(D4) `|Delta L_total|/L_SCALE` com `L_total = L_orb + L_spin`.** Sucessor de `INV-3`.

**Proibições explícitas de asserção.** Nenhuma execução com fusão ou fragmentação pode sustentar:
afirmação sobre conservação de energia mecânica; comparação numérica direta com
`results/2026/longrun_energy.csv`; uso de `INV-4` ou `INV-3` na forma de `integradores.md`;
classificação de integradores por comportamento energético. **A comparação entre integradores
continua sendo feita sem colisões.** As colisões são um estudo de física, não um banco de teste de
integradores; misturar os dois papéis anula os dois.

### 4.12 Crescimento descontrolado — há um teto, e ele é fechado

A preocupação é legítima e o cálculo a responde. Há dois laços de realimentação em sentidos opostos.

**Laço de fusão (positivo).** Fundir aumenta `M` e `R_sum`, o que aumenta
`v_esc_eff² = 2GM/sqrt(R_sum²+eps²)`, o que **baixa** `x`, o que aumenta `p_fus`. Além disso a
secção de choque com foco gravitacional `sigma = pi R_sum²(1 + 1/x)` cresce com `m`. No regime
dominado pelo softening (`R_sum << eps`) isso dá `dm/dt ∝ m^(5/3)`, que é **blowup em tempo finito**
— mais rápido que o `m^(4/3)` clássico de acreção de planetesimais. Sem oposição, um corpo semente
atinge `100 m_bar` em `0.35 t_ff`. **[T]** dado o modelo de taxa.

**Laço de fragmentação (negativo), e é ele que ganha.** Um corpo de massa `m` que colide com um de
massa `m_bar` e fragmenta é substituído pelo maior fragmento, de massa esperada `k (m + m_bar)`,
com

```
k = E[ max(f, 1-f) ] ,  f ~ U(a, 1-a)     ->     max(f,1-f) ~ U(1/2, 1-a)     ->     k = 3/4 - a/2

a = 0.1   =>   k = 0.700000  exatamente                                            [T]
```

A variação esperada de massa por evento é

```
E[Delta m] = p_fus * m_bar  +  p_frag * [ k (m + m_bar) - m ]
           = p_fus * m_bar  +  p_frag * [ k m_bar - (1-k) m ]
```

que se anula em

```
m* / m_bar  =  ( p_fus + k p_frag ) / ( (1-k) p_frag )                             [T]
```

**Este é um teto fechado, e é um ponto fixo atrator** (o termo `-(1-k) p_frag m` é linear e
negativo). Valores **[T]**:

| caso | `p_fus` | `p_frag` | `m*/m_bar` | `m*/M_tot` |
|---|---|---|---|---|
| controle uniforme `1/3` | `0.3333` | `0.3333` | `5.667` | `0.57%` |
| mapa padrão, `x` autoconsistente `= 1.450` | `0.2555` | `0.3274` | **`4.935`** | `0.49%` |

Integração da EDO de campo médio, com a taxa da Seção 4.1 e o mapa da Seção 4.7 **[T]**:

| `chi` | `m/m_bar` em `0.21 s` | em `0.63 s` | `x` final | `p_fus` | `p_frag` |
|---|---|---|---|---|---|
| `1` | `4.70` | `4.70` | `1.669` | `0.243` | `0.342` |
| `0.5` | `4.82` | `4.84` | `1.537` | `0.250` | `0.333` |
| `0.25` | `3.67` | `4.90` | `1.452` | `0.255` | `0.327` |
| `0.1` | `1.48` | `2.47` | `1.544` | `0.250` | `0.334` |

Sensibilidade: `v_coh = 2 V_CHAR` dá `m* = 6.78 m_bar`; `3 V_CHAR` dá `8.88`; `w = 1` dá `3.71`;
`w = 6` dá `5.26`. **[T]** O teto varia por fator `< 2.5` sobre variações de fator `3` nos
parâmetros.

**Veredito: NÃO há crescimento descontrolado com os parâmetros padrão. O maior corpo satura em
`~5 m_bar`, isto é `~0.5%` da massa total.** A causa é a fragmentação, não o mapa de regime — o teto
existe até no controle uniforme. **O parâmetro que de fato o controla é `a`** (o corte de `f`), via
`k = 3/4 - a/2`: `a -> 0` dá `k = 0.75` e teto maior; `a -> 0.5` dá `k = 0.5` e teto menor. Se o
runaway for desejado como objeto de estudo, é `a` que se mexe, não `w`.

Taxa de despovoamento, com `chi = 0.1` e `p_fus = 0.255`: `~56` fusões durante o primeiro rebote,
contra `500` corpos no núcleo — **`11%` do núcleo, `5.6%` da população**, por rebote. Com
`chi = 0.25` seriam `~348`, isto é, o núcleo inteiro em um rebote. **[T]** Este cálculo, e não uma
preferência estética, é o que fixa `chi = 0.1` como recomendação da Seção 4.1.

### 4.13 O invariante de ensemble

O critério de aceitação **não é** "não pode haver runaway" — runaway pode ser a física correta. O
critério é sobre a **degenerescência**: rejeita-se o modelo se toda semente terminar em um corpo, e
depressa.

**Protocolo (normativo).** `K_SEEDS = 32` execuções do estágio 3, `t_end = 3 t_ff`, variando apenas
`mass_seed` e a semente de colisão; posições e `Q` fixos. `K = 32` é justificado por custo
(`~38 s` por execução a `dt = 1.25e-4`, `~20 min` no total) e por resolução: com `0/32` seeds
degenerados, a regra de três limita a taxa verdadeira a `3/32 = 9.4%` a `95%` de confiança —
exatamente a resolução que o critério (C1) exige.

Grandezas registradas por semente: `N_final` (corpos vivos em `3 t_ff`), `t_50` (primeiro instante
com `N_live <= N/2`, ou `> 3 t_ff`), `n_merge`, `n_frag`, `n_elastic`, `max_i m_i / M_real`,
`min_i m_i / m_bar`, `min_t E_int/|E_0|`.

**Critérios de aceitação (`INV-31`):**

- **(C1) Não degenerado no ponto final.** `#{sementes com N_final = 1} / K <= 0.10`.
- **(C2) Não degenerado do outro lado.** mediana de `n_merge + n_frag + n_elastic >= 50`. Zero
  eventos é tão degenerado quanto todos.
- **(C3) Não rápido demais.** `t_50 > 1.0 t_ff` em pelo menos `90%` das sementes. Se a população
  cai pela metade antes de o colapso completar, o modelo de colisão substituiu a física que estava
  sendo demonstrada em vez de decorá-la.
- **(C4) Índice de dispersão.** `D := Var_sementes(n_merge) / media_sementes(n_merge)`. Exige-se
  `D <= 5`. Justificativa: para um processo de contagem bem comportado, `D ≈ 1` (Poisson). `D >> 1`
  é a assinatura de **bimodalidade** — algumas sementes disparam, outras não —, que é precisamente
  a degenerescência que o critério existe para pegar. `D` é o instrumento certo; a dispersão de
  `N_final` **não** é, porque com `~150` fusões esperadas o desvio relativo de `N_final` é
  `sqrt(150)/850 ≈ 1.4%`, e um critério de dispersão sobre `N_final` reprovaria um modelo saudável.
- **(C5) Mapa exercitado.** Cada canal recebe `>= 5%` dos eventos, agregado sobre o ensemble.

Todos os limiares acima são **[A]** e são critérios de **projeto**, não medições. Se o estágio 3
falhar (C1) ou (C3), a correção é reduzir `chi`; se falhar (C2), aumentar `chi`; se falhar (C5),
aumentar `w` ou reduzir `b`. Nenhum deles pode ser afrouxado depois de ver o resultado — ver a regra
do cabeçalho de `tests/tolerances.py`.

---

## 5. Slots mortos com `m = 0`

A fusão é `2 -> 1`. Para que as formas dos tensores nunca mudem — evitando recompilação de
`torch.compile` e do kernel Triton, e mantendo `a_current` válido na cadeia de reaproveitamento do
Verlet — o slot liberado recebe `m = 0` em vez de ser removido.

### 5.1 Verificação de inércia física, caminho por caminho

Verificado com o backend `torch_eager`, `eps = 0.05`, inserindo um slot morto **coincidente com a
partícula 0** (o caso mais adverso) **[M]**:

| caminho | resultado | veredito |
|---|---|---|
| força **sobre** os vivos | `max \|Δa_vivos\| = 0.0` — **bit a bit idêntica** | inerte |
| energia cinética `K` | idêntica bit a bit (`0.5 * 0 * \|v\|² = 0`) | inerte |
| energia potencial `U` | idêntica bit a bit (`m_i m_j = 0`) | inerte |
| energia total `E` | idêntica bit a bit | inerte |
| momento linear `P` | idêntico bit a bit (`torch.equal`) | inerte |
| momento angular `L` | idêntico bit a bit (`torch.equal`) | inerte |
| centro de massa | idêntico bit a bit; contribui `0` ao numerador **e** ao denominador | inerte |
| `half_mass_radius` | **quebrado já antes de `m = 0`** — ver 5.3 | **corrigir** |
| força **sobre** o slot morto | finita e não nula: `a = (0.390, -0.015, 0.595) m/s²` | inofensiva |

A inércia é exata porque `m_j` multiplica cada termo de par: `minv = m.unsqueeze(0) * inv` no kernel
de aceleração, e `m_i m_j inv_sqrt` no potencial. **[T]**

O slot morto **sofre** aceleração e portanto se move. Isso é inofensivo (ele não contribui a nada),
mas significa que ele não fica onde foi parado.

### 5.2 O único ponto onde `m = 0` produz `NaN`

**Com `eps = 0.0` e um slot morto coincidente com um corpo vivo, o campo de aceleração vira `NaN`.**
Verificado: `n_nan = 6` (as duas linhas envolvidas, 3 componentes cada). **[M]**

Mecanismo: fora da diagonal, `dsq = |diff|² + eps² = 0`, logo `inv = dsq^(-1.5) = inf`, e
`minv = 0 * inf = nan`, que propaga. O mascaramento da diagonal não ajuda: `i != j`. Deslocando o
slot morto em `1e-3 m`, o `NaN` desaparece. **[M]**

Com `eps = 0.05` **não há `NaN` em caminho algum** — `dsq >= eps² > 0` sempre.

**Requisitos normativos:**

1. **Colisões exigem `eps > 0`.** `integrate(..., collision=...)` com `softening == 0.0` deve
   levantar `ValueError` nomeando a incompatibilidade. Não é uma restrição onerosa: `INV-5`
   (Kepler, `eps = 0`) é um teste de dois corpos sem colisões, e não há caso de uso legítimo para
   colisão sem regularização.
2. **`r` e `v` do slot morto:** `r_morto = r_fundido`, `v_morto = v_fundido`. Justificativa: garante
   finitude, garante que o slot morto nunca ocupe região inacessível a um corpo vivo (o que
   corromperia diagnósticos de caixa envolvente ou `max_i |r_i|`), e mantém os dois co-movendo-se.
   Parqueá-lo longe é **proibido**: infla `max_i |r_i|` e pode transbordar em fp32. Zerar `r` o
   empilha na origem, o que é aceitável mas menos informativo.
3. **`INV-1` (`a_i = -(1/m_i) ∂U/∂r_i`) é indefinido para `m_i = 0`** e deve ser verificado apenas
   sobre slots vivos.
4. **`N` nas cotas de arredondamento** (`TOL-CENTER`, `TOL-MOM`) continua sendo o **comprimento do
   array**, não a contagem de vivos: a redução ainda soma `N` parcelas, e é o número de parcelas que
   governa o erro.

### 5.3 `half_mass_radius` está errado, e já estava

**Confirmação do apontamento.** `src/nbody/observables.py:68-74` devolve `sorted_d[n // 2 - 1]`: a
mediana de **contagem**. Isso é a mediana de **massa** apenas quando as massas são iguais **e** `N` é
par. Verificado **[M]**:

| caso | mediana de contagem | mediana de massa | iguais? |
|---|---|---|---|
| `N = 1000`, massas iguais | `1.524991065099` | `1.524991065099` | **sim, bit a bit** |
| `N = 6`, massas iguais | `1.222234377749` | `1.222234377749` | sim |
| `N = 999`, massas iguais | `1.527324083263` | `1.528571815588` | **não** |
| `N = 5`, massas iguais | `1.181311609443` | `1.379116600854` | **não** |
| `N = 1000`, `m_max/m_min = 223` | `1.583022` | `1.434636` | **não (10%)** |

**Versão correta (normativa):**

```
c   = centro de massa
d_i = |r_i - c|
pi  = permutacao que ordena d crescente
C_k = sum_{l <= k} m_{pi(l)}                       (massa acumulada)
k*  = min { k : C_k >= M_real / 2 }
r_half = d_{pi(k*)}
```

Três propriedades **[T]**:

1. **Reduz-se exatamente à fórmula atual** para massas iguais e `N` par: `C_k = k m`, e
   `C_k >= N m/2` equivale a `k >= N/2`, logo `k* = N/2` em base 1, isto é, índice `N//2 - 1` em
   base 0. Confirmado bit a bit para `N = 1000`. **Nenhum valor publicado muda**:
   `IC_R_HALF_0 = 4.881251` e `COLLAPSE_R_HALF_MIN = 0.3472` permanecem válidos.
2. **Slots mortos caem fora sozinhos**, sem caso especial: contribuem `0` à massa acumulada.
3. **`k*` nunca é um slot morto.** Se `pi(k*)` tivesse `m = 0`, então `C_{k*} = C_{k*-1} >= M/2`,
   contradizendo a minimalidade de `k*`. **[T]**

Para `N` ímpar as duas fórmulas divergem; o projeto usa `N = 1000`, então nenhum resultado
existente é afetado. A correção é obrigatória mesmo assim: assim que as massas deixam de ser iguais,
a fórmula atual mede a mediana errada — e é justamente com massas desiguais que o observável será
usado.

---

## 6. Invariantes testáveis (`INV-11` em diante)

Cada invariante traz enunciado, procedimento, tolerância e o que sua falha significa. Numeração
continua a de `integradores.md`; `INV-1` a `INV-10` permanecem em vigor, com as emendas da Seção 9.

### `INV-11` — A amostra de massas segue a lei de potência truncada

**Enunciado.** As massas sorteadas seguem `p(m) ∝ m^(-alpha)` em `[m_min, m_max]`.

**Procedimento.** Agregar `M = 200` realizações (`2e5` massas). Testar por Kolmogorov–Smirnov contra
a CDF analítica da Seção 2.2 **restrita ao setor apropriado**: as massas do corpo contra `F` truncada
a `[m_min, m_big]`, as da cauda contra `F` truncada a `[m_big, m_max]`. Testar as duas separadamente,
**não** a mistura — a mistura não é a lei não condicionada (Seção 2.8) e um teste KS sobre ela falha
contra uma implementação correta.

**Tolerância.** `p_KS >= 0.01` em cada setor. Justificativa: com `n ≈ 2e5` amostras a estatística KS
tem cauda muito fina; `0.01` produz falso positivo em `1%` das execuções da suíte, o que é aceitável
para um teste marcado como estocástico, e detecta desvios de forma `>~ 0.5%` na CDF. Semente do teste
fixa, de modo que a execução é determinística apesar de o critério ser estatístico.

**Teste complementar, determinístico.** `m_min` e `m_max` da amostra devem estar dentro dos limites,
e a média amostral de `2e5` massas deve satisfazer `|<m>_amostra / <m>_teorico - 1| <= 0.05`, com
`<m>_teorico` sendo o valor **condicionado** `9.9239075e11/1000` (Seção 2.8) e a cota de `5%`
derivada de `3 * CV / sqrt(200) = 3 * 0.0924 / 14.14 = 0.0196`, arredondada para cima com margem
`2.5x`.

**Se falhar.** Expoente errado na CDF inversa (o mais comum: `1/(1-alpha)` escrito como
`1/(alpha-1)`), ou ramo `alpha = 1` ausente. **Bloqueante.**

### `INV-12` — Contagem de corpos massudos e sua distribuição

**Enunciado (a), determinístico.** Em **toda** realização, `1 <= #{i : m_i > m_big} <= 3`.

**Procedimento (a).** `1000` realizações com sementes distintas. Critério: **zero** violações. Não é
estatístico — é uma propriedade estrutural do amostrador.

**Enunciado (b), estatístico.** A distribuição empírica de `k` sobre `1000` realizações concorda com
a binomial renormalizada `(0.37476530, 0.37514082, 0.25009388)`.

**Tolerância (b).** Teste qui-quadrado com `2` graus de liberdade, `p >= 0.01`. Equivalentemente,
cada `|f_k - P_k| <= 3 sqrt(P_k(1-P_k)/1000)`, isto é, `+-0.046`, `+-0.046`, `+-0.041`. Derivada do
erro-padrão binomial, não de saída observada.

**Se falhar (a).** Renormalização ausente (então `k = 0` aparece em `13.5%` das realizações e
`k >= 4` em `14.3%`). **Bloqueante.**

**Se falhar (b) mas não (a).** Pesos renormalizados errados — tipicamente uniformes em `{1,2,3}`,
que daria `(1/3, 1/3, 1/3)` e falharia o qui-quadrado com folga.

### `INV-13` — Permutação uniforme dos slots massudos

**Enunciado.** A posição do slot de cada corpo massudo é uniforme em `{0, ..., N-1}`.

**Procedimento.** `4000` realizações; registrar o índice do slot de massa máxima. Dividir
`[0, N)` em `10` caixas e aplicar qui-quadrado com `9` graus de liberdade contra uniforme.

**Tolerância.** `p >= 0.01`. Com `4000` amostras em `10` caixas (`400` esperadas por caixa), o teste
detecta um desvio de `>~ 10%` na frequência de uma caixa.

**Se falhar.** O passo 4 da Seção 2.7 foi omitido. A amostra continua correta como multiconjunto,
mas a lei conjunta `(posição, massa)` não é a condicional (Seção 2.6) e todo estudo de ensemble sobre
a posição inicial dos massudos está viciado. **Bloqueante** — e é um modo de falha que **nenhum**
outro invariante deste documento pega.

### `INV-14` — Massa média e forma fechada de `m_min`

**Enunciado.** `<m>` da lei não condicionada é `PARTICLE_MASS` por construção de `m_min`.

**Procedimento.** Quadratura numérica de `∫ m p(m) dm` sobre `[m_min, m_max]` com `m_min` produzido
pela implementação, **sem** reusar a fórmula fechada de `g(alpha, R)`.

**Tolerância.** `|<m>_quad / PARTICLE_MASS - 1| <= 1e-10` em fp64. Justificativa: a quadratura de
uma função suave em `[m_min, m_max]` com `>= 1e4` nós de Simpson tem erro de truncamento muito abaixo
disso; a cota é dominada pelo arredondamento acumulado sobre `1e4` termos, `~1e4 * eps_fp64 = 2e-12`,
com margem `50x`. Executar também com `alpha ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}` — os dois casos
degenerados **têm** de estar na lista.

**Se falhar em `alpha = 1` ou `alpha = 2` apenas.** Ramo degenerado ausente (Seção 2.2/2.3).

### `INV-15` — `t_ff` da massa realizada

**Enunciado.** `t_ff` é calculado de `M_real = sum_i m_i`, não de `N * PARTICLE_MASS`.

**Procedimento (a), determinístico.** Construir um estado com massas escaladas por um fator `2.0`
exato e verificar `t_ff(2M)/t_ff(M) = 2^(-1/2)` com erro relativo `<= 1e-14` (fp64). Isto testa a
dependência funcional sem depender de sorteio.

**Procedimento (b), de ensemble.** Sobre `200` realizações, `sd(t_ff)/media(t_ff)` deve cair em
`[0.030, 0.065]`. Justificativa: a predição é `4.62%` (Seção 2.10); a banda é `+-35%` em torno dela,
que é `~5` erros-padrão da própria estimativa de desvio-padrão com `200` amostras
(`sd de sd ≈ sd/sqrt(2n) = 5%`).

**Se falhar (a).** `T_FF` constante usada em vez do valor por realização — o erro mais consequente e
menos visível de toda esta especificação, porque produz um eixo temporal errado em até `5%` sem
qualquer sintoma numérico.

### `INV-16` — Razão virial realizada, momento nulo, teto e não degenerescência

**Enunciado (a).** `|2K/|U| / Q - 1| <= 1e-12` em fp64. Exato por construção do `lambda`
(Seção 3.3); a cota cobre o arredondamento de uma redução de `N` termos.

**Enunciado (b).** `||P|| / (M_real * v_rms) <= N * eps_prec`. Mesma estrutura de `TOL-MOM`: a
subtração da média zera `P` em aritmética exata e o resíduo é de arredondamento (Seção 5.5 de
`integradores.md`).

**Enunciado (c).** `max_i |v_i| <= lambda * f_cut * v_esc`, com `lambda` reportado. A implementação
**deve** expor `lambda`; um teste que exija `max_i |v_i| <= f_cut * v_esc` sem `lambda` falha contra
implementação correta em cerca de um terço das sementes (Seção 3.3.1).

**Enunciado (d), isotropia.** Sobre `50` realizações agregadas, o tensor
`sum_i m_i v_i v_i^T / sum_i m_i |v_i|²` tem autovalores em `1/3 +- 3 sqrt(2/(45 N M_real_eff))`.
Forma operacional simples e suficiente: cada autovalor em `[0.30, 0.37]` para `N = 1000` agregado
sobre `50` realizações. Derivada da variância de um estimador de segundo momento gaussiano, não de
saída observada.

**Enunciado (e), não degenerescência.** Com `Q > 1.532168 * f_cut^2`, `ValueError`. Testar em
`(Q, f_cut) = (0.5, 0.5)` — o caso vetado na Seção 3.4 — e verificar que a exceção nomeia o `Q`
máximo.

**Se falhar (d).** Rejeição por componente em vez de por rapidez (Seção 3.3, passo 1). É o modo de
falha que passa por todos os outros testes desta lista.

### `INV-17` — `Q = 0` reproduz o colapso frio bit a bit

**Enunciado.** Com `Q = 0.0` e massas iguais, o `State` devolvido é **bit a bit idêntico** ao de
`cold_sphere` com a mesma semente.

**Procedimento.** `torch.equal` em `r`, `v` e `m`. Verificar também que `v` não contém `-0.0`
(`torch.signbit(v).any() == False`), e que o `Generator` de velocidades não foi consumido.

**Tolerância.** Binária. Sem folga.

**Se falhar.** O curto-circuito da Seção 3.6 está ausente e o caminho `lambda = 0` foi tomado,
produzindo `-0.0` — o que quebra a verificação bit a bit contra
`data/ic_sphere_N1000_seed20190222.npz`. **Bloqueante.**

### `INV-18` — Álgebra da detecção varrida e ausência de tunelamento

**Enunciado (a).** `t*` é o minimizador de `|dr + t dv|²` em `[0, h]`.

**Procedimento (a).** `200` configurações aleatórias, incluindo casos que ativam cada extremo do
clamp e o caso `|dv|² = 0`. Comparar com minimização por varredura de `1e5` pontos.

**Tolerância (a).** `|t*_formula - t*_grade| <= h/1e5` e
`| |sep|_formula / |sep|_grade - 1 | <= 1e-12`. Justificativa: o erro da grade é `O(h/1e5)` por
construção; a segunda cota é de arredondamento porque a parábola é plana no mínimo.

**Enunciado (b), Courant colisional.** `max C_coll = max |u| dt / (R_i+R_j) <= 1` sobre todos os
candidatos, ao longo de toda a execução.

**Tolerância (b).** `<= 1.0`, sem margem. Não é uma tolerância numérica: é a condição de validade do
detector (Seção 4.4). `C_coll` medido deve ser gravado no CSV de cada execução.

**Se falhar (b).** `dt` grande demais. Reduzir para `DT_COLLISION` ou menos. **Invalida os
resultados colisionais da execução**, e não apenas os degrada.

### `INV-19` — Pareamento disjunto: massa, determinismo, e taxa de rejeição

**Enunciado (a).** `|sum_i m_i (depois) / sum_i m_i (antes) - 1| <= n_events * eps_prec`, por passe.
Derivada de um arredondamento por evento (Seções 4.5 e 4.9).

**Enunciado (b).** Nenhum slot participa de dois eventos no mesmo passe. Teste direto sobre a lista
de eventos aceitos.

**Enunciado (c).** Determinismo: duas execuções com as mesmas sementes produzem listas de eventos
idênticas, na mesma ordem, incluindo em caso de `t*` empatado. Construir deliberadamente um caso
com dois pares de `t*` idêntico bit a bit (configuração simétrica) e verificar reprodutibilidade.

**Enunciado (d).** `f_reject <= 0.05` sobre toda a execução.

**Se falhar (c).** Chave de ordenação sem desempate por `(i, j)` — o resultado deixa de ser
reprodutível entre dispositivos e a execução não é publicável.

**Se falhar (d).** Multiplicidade de eventos alta demais para a aproximação gulosa; reduzir `dt` ou
`chi`.

### `INV-20` — Colisão elástica: `m`, `P`, `L`, `K` exatos

**Procedimento.** `500` eventos sintéticos com massas, posições e velocidades aleatórias
(razão de massa até `1000`), aplicando o mapa da Seção 4.9 isoladamente.

**Tolerâncias**, todas normalizadas e derivadas de arredondamento de reduções curtas (`~10` termos),
com margem `~100x` sobre o medido:

| grandeza | cota |
|---|---|
| `\|Δm\|/M` | `<= 4 eps_prec` |
| `\|ΔP\|/\|P\|` | `<= 100 eps_prec` (medido `5.06e-17`) |
| `\|ΔL\|/\|L\|` | `<= 100 eps_prec` (medido `3.63e-17`) |
| `\|ΔK\|/K` | `<= 100 eps_prec` (medido `0`) |
| `\|Δ sum m r\|/\|sum m r\|` | `<= 100 eps_prec` (medido `0`) |

**Se `ΔL` falhar mas `ΔK` passar.** O impulso foi aplicado com normal não paralela à separação **no
ponto de aplicação** — quase certamente porque a colisão foi resolvida no fim do passo em vez de em
`t*` (Seção 4.5). `K` não detecta esse erro, porque a reflexão conserva `K` para **qualquer** normal
unitária. Este é o par de testes mais discriminante da suíte de colisões.

### `INV-21` — Fusão: conservações exatas e destruições exatas

**Tolerâncias** (mesma base de `INV-20`):

| grandeza | cota | observação |
|---|---|---|
| `\|Δm\|/M` | `<= 4 eps_prec` | |
| `\|ΔP\|/\|P\|` | `<= 100 eps_prec` | medido `0` |
| `\|Δ sum m r\|/\|sum m r\|` | `<= 100 eps_prec` | medido `0` |
| `\| ΔK / (-T_cm) - 1 \|` | `<= 1e-12` | medido `1.01e-15`; **destruição exata e prevista** |
| `\| \|ΔL\| / \|mu (dr x u)\| - 1 \|` | `<= 1e-12` | **destruição exata e prevista** |

**Comutação com o drift.** Verificar que fundir-e-depois-derivar e derivar-e-depois-fundir produzem
`sum_i m_i r_i` idêntico dentro de `100 eps_prec` (Seção 4.9). **[T]**

**Se `ΔL` for zero.** A implementação está conservando `L` que não deveria conservar — provavelmente
somando um termo de spin diretamente em `L_orb` em vez de em `L_spin`, o que mascara o efeito e
torna `INV-24` vazio.

### `INV-22` — Fragmentação: `m`, `P`, `T_cm`, `K` exatos

**Tolerâncias:**

| grandeza | cota | medido |
|---|---|---|
| `\|Δm\|/M` | `<= 4 eps_prec` | `0` |
| `\|ΔP\|/\|P\|` | `<= 100 eps_prec` | `5.06e-17` |
| `\|ΔT_cm\|/T_cm` | `<= 100 eps_prec` | `0` |
| `\|ΔK\|/K` | `<= 100 eps_prec` | `3.43e-16` |
| `\|Δ sum m r\|/\|sum m r\|` | `<= 100 eps_prec` | `2.19e-17` |
| `\| \|u'\|/\|u\| / sqrt(mu/mu') - 1 \|` | `<= 1e-12` | concordância a 6 casas |

**Isotropia da direção de `u'`.** `5000` eventos sintéticos; o tensor de segundo momento de
`u'/|u'|` tem autovalores em `[0.30, 0.37]` — mesma derivação de `INV-16(d)`.

**Razão de massa.** `f ∈ [0.1, 0.9]` em todos os eventos (determinístico), e a distribuição de
`f` é uniforme por KS (`p >= 0.01`).

### `INV-23` — `E_total = K + U + E_int` e o sinal de `E_int`

**Enunciado (a), por evento.** Recalculando `K + U + E_int` imediatamente antes e depois de cada
mapa, com posições congeladas: `|Δ(K+U+E_int)| / |E_0| <= TOL-EVENT-CONS`.

**Enunciado (b), ao longo da execução.** `|ΔE_total/E_total(0)|` obedece aos critérios
**qualitativos** de `INV-4`, transferidos de `E_mec` para `E_total`:
`velocity_verlet` não monótono e com final `<= pico/10`; `euler` monótono crescente com final
`>= +0.3`; `rk4` com final negativo e `>= pico/3`. **Os valores `[M]` de `integradores.md` não se
transferem** — a trajetória é outra.

**Enunciado (c), sinal.** `min_t E_int(t) >= -1e-3 * |E_0|`.

**Se (a) falhar.** Termo de terceiro corpo omitido em `ΔU`, ou sinal trocado no acumulador. É o
teste que pega o erro que `E_total` conservada, sozinha, esconde por construção.

**Se (c) falhar.** O modelo está injetando energia mecânica (Seção 4.10). Mitigação declarada:
`eta > 0` na fragmentação. **Não** afrouxar a cota.

### `INV-24` — `L_total = L_orb + L_spin`

**Enunciado.** `|ΔL_total| / L_SCALE <= 1e-12` em fp64 ao longo de `RUN_COLLISION` com
`velocity_verlet`, com `L_SCALE = M_real * R_0 * V_CHAR` recalculado da massa realizada.

**Enunciado complementar, de poder discriminante.** `|L_spin(t_end)| / L_SCALE` deve ser **não
desprezível** — cota inferior `1e-8`. Se `L_spin` permanecer no nível de arredondamento, ou os
eventos não estão ocorrendo (então `INV-31(C2)` também falha), ou o termo de spin não está sendo
acumulado, e `INV-24` estaria passando vazio.

### `INV-25` — O mapa de regime é bem formado

**Enunciado.** Sobre `x ∈ [1e-6, 1e6]` em `601` pontos logarítmicos, e para
`w ∈ {0.5, 1, 3, 5, 100, inf}`:

1. `|p_fus + p_el + p_frag - 1| <= 4 eps_prec`;
2. `p_c > 0` estritamente, para todo `c` e todo `x`, inclusive nos clamps;
3. `p_fus` não crescente e `p_frag` não decrescente em `x`, dentro de `1e-15` por passo;
4. `p_el` máxima em `x = 1`;
5. `w = inf` (e `w = 1e6`) dá `|p_c - 1/3| <= 1e-5` para todo `c` e todo `x`;
6. nos extremos `x = 1e-300` e `x = 1e300`, `min_c p_c >= 1e-30` — verificado `8.76e-27`, acima do
   menor normal de fp32 (`1.18e-38`).

**Tolerâncias.** Todas de arredondamento ou estruturais; nenhuma ajustada a saída.

**Se (2) ou (6) falhar.** *Log-sum-exp* ou clamp ausentes. Um canal com probabilidade exatamente
zero viola o requisito explícito do projeto.

### `INV-26` — Consistência entre o mapa e o sorteador

**Enunciado.** As frações de canal realizadas concordam com as previstas pela integração do mapa
sobre o histograma de `x` efetivamente visitado:

```
| f_c - <p_c> |  <=  3 * sqrt( <p_c (1 - p_c)> / n_events )
```

com `<·>` a média sobre os eventos registrados. Esta é a cota binomial a `3` desvios-padrão, derivada
e não ajustada.

**Procedimento.** Registro de eventos do estágio 3, agregado sobre as `32` sementes.

**Critério adicional (calibração).** Cada `f_c >= 0.05`. Se falhar, aumentar `w` ou reduzir `b`
(Seção 4.13, C5) — **antes** de olhar qualquer outro resultado.

**Se falhar a cota binomial.** O `x` usado no sorteio não é o `x` registrado — tipicamente `E_bind`
computado com o potencial não suavizado num lugar e suavizado no outro.

### `INV-27` — `m = 0` é fisicamente inerte

**Enunciado.** Inserir um slot com `m = 0` **coincidente com um corpo vivo** não altera, **bit a
bit**, nenhum de: campo de aceleração sobre os vivos, `K`, `U`, `E`, `P`, `L`, centro de massa,
`half_mass_radius`.

**Procedimento.** `torch.equal` em cada caso, com `eps = 0.05`. Repetir com o slot morto deslocado e
com vários slots mortos.

**Tolerância.** Binária. Sem folga — a inércia é exata (Seção 5.1), não aproximada.

**Se falhar.** Alguma redução não multiplica por `m`. `half_mass_radius` falha se a correção da
Seção 5.3 não tiver sido aplicada.

### `INV-28` — `eps = 0` com colisões é recusado

**Enunciado.** `integrate(..., softening=0.0, collision=<ligada>)` levanta `ValueError`.

**Procedimento complementar, de demonstração.** Verificar diretamente que
`accelerations` com `eps = 0` e um slot morto coincidente produz `NaN` (medido: `6` entradas). Este
teste documenta **por que** a recusa existe e falha ruidosamente se alguém "consertar" o kernel de
um jeito que mascare o problema em vez de removê-lo.

### `INV-29` — `half_mass_radius` é a mediana de massa

**Enunciado (a), compatibilidade.** Para massas iguais e `N` par, o valor é **bit a bit idêntico** ao
da fórmula de contagem. Verificar em `N ∈ {6, 100, 1000}`.

**Enunciado (b), correção.** Para massas desiguais, o valor é a mediana de massa. Verificar contra
uma implementação de referência independente escrita no teste (ordenar, acumular, procurar). Com
`m_max/m_min = 223` as duas diferem em `~10%` — separação larga o bastante para não haver ambiguidade.

**Enunciado (c), slots mortos.** Inserir slots mortos não altera o valor (coberto por `INV-27`), e
`k*` nunca cai sobre um slot morto.

**Tolerância.** (a) binária; (b) `<= 1e-14` relativo contra a referência do teste; (c) binária.

### `INV-30` — Equivalência com colisões desligadas

**Enunciado.** Com `R_ref = 0.0` (ou `collision=None`), `integrate` produz trajetória **bit a bit
idêntica** à da versão sem o parâmetro, para os quatro integradores.

**Procedimento.** `RUN_COLLAPSE` reduzido (`n_steps = 200`), `torch.equal` em `r` e `v`.

**Tolerância.** Binária.

**Se falhar.** O caminho de colisão está alterando o estado mesmo sem eventos — tipicamente um
reordenamento de operações no drift. **Bloqueante:** sem este invariante nenhuma comparação entre
execuções com e sem colisão tem sentido, porque as diferenças não seriam atribuíveis às colisões.

### `INV-31` — Ensemble: não degenerescência

Os critérios (C1) a (C5) da Seção 4.13, mais:

**(C6) Moinho de fragmentação.** `min_i m_i / m_bar >= 1e-3` em pelo menos `90%` das sementes, e o
número de corpos abaixo de `m_min` é reportado. **[A]** — a cota `1e-3` é de projeto: abaixo dela o
raio de contato caiu por `10x` e o corpo é efetivamente não colisional, de modo que o moinho se
extingue sozinho; o critério existe para tornar isso visível, não para impedi-lo.

**(C7) Detecção de runaway.** `t_runaway :=` primeiro `t` com `max_i m_i / M_real >= 0.10`, ou
`> 3 t_ff`. Deve ser reportado e **marcado em toda figura**. Predição a confrontar: pelo modelo de
campo médio da Seção 4.12, `max_i m_i / M_real` satura em `~0.005`, isto é, `20x` abaixo do
limiar — logo `t_runaway > 3 t_ff` é o resultado esperado. **[T]** dado o modelo; **[A]** quanto ao
sistema real. Se o runaway ocorrer, ele é o resultado e deve ser relatado como tal; o que não é
permitido é que ocorra sem ser detectado, porque a partir daí `V_CHAR`, `L_SCALE` e a própria
escolha de `eps` deixam de descrever o sistema.

---

## 7. Tolerâncias

Todas as tolerâncias abaixo são **relativas e adimensionais**, conforme a regra sem exceção da Seção
6.3 de `integradores.md`. Nenhuma foi ajustada a saída observada; cada uma traz a origem.

| identificador | grandeza | fp64 | fp32 | origem |
|---|---|---|---|---|
| `TOL-MASS-SUM` | `\|Δ sum_i m_i\| / M_real` por passe | `n_events * eps_prec` | idem | um arredondamento por evento (4.5, 4.9) |
| `TOL-EVENT-CONS` | `\|Δ(K+U+E_int)\| / \|E_0\|` através de um mapa de desfecho | `1e-13` | `1e-5` | redução de `O(N)` termos: `N eps_prec = 2.2e-13` fp64; margem `~2x`. fp32 herda `TOL-ENERGY` |
| `TOL-EVENT-INV` | conservações exatas por desfecho (`INV-20/21/22`) | `100 eps_prec` | `100 eps_prec` | reduções de `~10` termos; margem `~100x` sobre o medido (`5e-17`) |
| `TOL-EVENT-PRED` | destruições previstas (`ΔK = -T_cm`, `ΔL = -mu dr x u`) | `1e-12` | não testável | medido `1.01e-15`; margem `~1000x` |
| `TOL-VIRIAL` | `\|2K/\|U\| / Q - 1\|` | `1e-12` | `1e-5` | exato por construção; cobre redução de `N` termos |
| `TOL-COURANT` | `max C_coll` | `<= 1` | `<= 1` | condição de validade do detector, não tolerância numérica (4.4) |
| `TOL-REJECT` | `f_reject` do pareamento guloso | `<= 0.05` | idem | critério de projeto **[A]** (4.5) |
| `TOL-EINT-NEG` | `min_t E_int / \|E_0\|` | `>= -1e-3` | idem | critério de projeto **[A]** (4.10) |
| `TOL-PROB` | `\|sum_c p_c - 1\|` | `4 eps_prec` | `4 eps_prec` | três somas e uma divisão |
| `TOL-PROB-UNIF` | `\|p_c - 1/3\|` com `w = inf` | `1e-5` | `1e-5` | exato; a cota cobre `b/w` residual |
| `TOL-CHANNEL` | `\|f_c - <p_c>\|` | cota binomial `3 sigma` | idem | erro-padrão binomial (`INV-26`) |
| `TOL-HALF-MASS` | `r_half` contra referência do teste | `1e-14` | não aplicável | comparação de dois valores ordenados |

`eps_prec ∈ {eps_fp64 = 2.220446e-16, eps_fp32 = 1.1920929e-07}`, como em `integradores.md`.

**Notas normativas.**

- **`TOL-EVENT-PRED` não existe em fp32 deliberadamente.** A concordância de `1e-15` entre `ΔK` e
  `-T_cm` é uma identidade algébrica exata; em fp32 o piso é `~1e-7` e o teste mede arredondamento,
  não a identidade. O mesmo raciocínio da Seção 6.4 de `integradores.md`.
- **O diagnóstico de energia colisional (`K`, `U`, `E_int`) é sempre acumulado em fp64**, mesmo com
  núcleo em fp32, pela diretiva da Seção 6.4 de `integradores.md`. `E_int` é um acumulador de longo
  prazo sobre milhares de incrementos de sinais opostos; acumulá-lo na precisão do núcleo destruiria
  a medição que ele existe para fazer.
- **Nenhuma tolerância deste documento pode ser afrouxada após ver o resultado.** Onde um critério é
  de projeto e não derivado, ele está marcado **[A]** e a resposta a uma falha é mudar o
  **parâmetro** (`chi`, `w`, `b`, `dt`), não a cota.

---

## 8. Constantes para a implementação

```
# --- espectro de massas (Secao 2)
MASS_ALPHA            = 2.35             # expoente de Salpeter
MASS_RATIO            = 1000.0           # m_max / m_min
MASS_G_FACTOR         = 3.5136877959     # g(alpha, R) = <m>/m_min, forma fechada
MASS_MIN              = 2.8460126741e8   # kg = PARTICLE_MASS / MASS_G_FACTOR
MASS_MAX              = 2.8460126741e11  # kg = MASS_RATIO * MASS_MIN
MASS_BIG              = 2.7509063196e10  # kg = F^-1(1 - 2/N),  = 27.509 * PARTICLE_MASS
MASS_TAIL_PROB        = 2.0e-3           # p = 2/N
MASS_K_WEIGHTS        = (0.37476530, 0.37514082, 0.25009388)   # binomial renormalizada em {1,2,3}
MASS_K_ACCEPT_PROB    = 0.72223972       # P(K in {1,2,3}) -- fracao da medida retida
MASS_COND_MEAN_BIAS   = -7.609250e-3     # (E[M_tot|cond] - N<m>) / (N<m>)
MASS_COND_CV          = 9.2402e-2        # CV(M_tot | K in {1,2,3})
MASS_TFF_CV           = 4.6201e-2        # = MASS_COND_CV / 2
MASS_SEED             = 20190223         # fluxo SEPARADO do de posicoes (SEED = 20190222)

# --- velocidades (Secao 3)
Q_DEFAULT             = 0.25             # razao virial 2K/|U|
F_CUT_DEFAULT         = 0.5              # teto de truncamento em unidades de v_esc
Q_SUP_COEFF           = 2.009056         # Q_sup(f) = Q_SUP_COEFF * f^2   (limite degenerado)
Q_USABLE_COEFF        = 1.532168         # Q usavel  = Q_USABLE_COEFF * f^2  (x_c >= 2)
X_C_MIN               = 2.0              # truncamento nao mais apertado que 2 sigma
VEL_SEED              = 20190224         # terceiro fluxo, separado

# valores derivados para massas iguais (M_real = 1e12, |U_0| = 6.4260397026e12)
V_ESC_SPHERE          = 4.6386556804     # m/s = sqrt(2 G M / R_0) = 1.414229 * V_CHAR
VEL_CUT_DEFAULT       = 2.319328         # m/s = F_CUT_DEFAULT * V_ESC_SPHERE
VEL_SIGMA_DEFAULT     = 0.7604389        # m/s, raiz de sigma^2 h(v_cut/sigma) = <v^2>
VEL_XC_DEFAULT        = 3.0500           # = VEL_CUT_DEFAULT / VEL_SIGMA_DEFAULT
VEL_RMS_DEFAULT       = 1.267482         # m/s
VEL_MODE_DEFAULT      = 1.0754           # m/s = sqrt(2) * sigma  (rapidez MAIS provavel)
VEL_REJECT_FRACTION   = 2.5529e-2
VEL_LAMBDA_SD         = 1.291e-2         # sd de lambda = 0.5*sqrt(2/(3N))

# --- colisoes (Secao 4)
CHI_DEFAULT           = 0.1              # [A] R_ref = CHI_DEFAULT * SOFTENING
R_REF_DEFAULT         = 5.0e-3           # m
DT_COLLISION          = 1.25e-4          # s = DT_COLLAPSE / 4  (criterio C_coll < 1)
N_STEPS_COLLISION     = 50400            # 3 t_ff a DT_COLLISION (massas iguais)
COH_VELOCITY_FACTOR   = 1.0              # [A] v_coh = COH_VELOCITY_FACTOR * V_CHAR
MAP_B                 = 1.0986123        # [A] = ln 3, meia-largura do plato elastico em ln x
MAP_W                 = 3.0              # [A] largura de suavizacao; inf -> uniforme
MAP_S_CLAMP           = 30.0             # clamp de s/w; menor prob = e^-60 = 8.76e-27
FRAG_F_MIN            = 0.1              # f ~ U(FRAG_F_MIN, 1 - FRAG_F_MIN)
FRAG_ETA              = 0.0              # fracao dissipada na fragmentacao (desligada)
FRAG_K_MAX            = 0.70             # E[max(f,1-f)] = 3/4 - FRAG_F_MIN/2, EXATO
MASS_CAP_UNIFORM      = 5.667            # m*/m_bar no controle uniforme 1/3
MASS_CAP_DEFAULT      = 4.935            # m*/m_bar com o mapa padrao
COLLISION_SEED        = 20190225         # quarto fluxo, separado

# --- ensemble (Secao 4.13)
K_SEEDS               = 32
ENS_N_FINAL_1_MAX     = 0.10             # (C1)
ENS_MIN_EVENTS_MEDIAN = 50               # (C2)
ENS_T50_MIN_TFF       = 1.0              # (C3)
ENS_T50_FRACTION      = 0.90             # (C3)
ENS_DISPERSION_MAX    = 5.0              # (C4) Var(n_merge)/mean(n_merge)
ENS_CHANNEL_MIN       = 0.05             # (C5)
ENS_MIN_MASS_FLOOR    = 1.0e-3           # (C6) em unidades de m_bar
ENS_RUNAWAY_THRESHOLD = 0.10             # (C7) max_i m_i / M_real

# --- tolerancias novas (Secao 7)
TOL_EVENT_CONS_FP64   = 1e-13
TOL_EVENT_CONS_FP32   = 1e-5
TOL_EVENT_INV_ULP     = 100.0            # multiplica eps_prec
TOL_EVENT_PRED        = 1e-12
TOL_VIRIAL_FP64       = 1e-12
TOL_VIRIAL_FP32       = 1e-5
TOL_COURANT_MAX       = 1.0
TOL_REJECT_MAX        = 0.05
TOL_EINT_NEG          = -1e-3            # em unidades de |E_0|
TOL_PROB_UNIF         = 1e-5
TOL_HALF_MASS         = 1e-14
```

**Grandezas que deixam de ser constantes.** Com o espectro de massas ligado, os seguintes valores de
`integradores.md` §9 passam a ser **funções da realização** e não podem mais ser lidos de `config`:
`TOTAL_MASS`, `DENSITY`, `T_FF`, `T_CONV`, `V_CHAR`, `L_SCALE`, `U_MIN_BOUND`. A implementação deve
expô-los como funções do `State` (por exemplo `scales_from_state(state)`), mantendo as constantes de
módulo apenas para o caso de massas iguais. Ler `config.T_FF` numa execução com massas sorteadas é
o erro que `INV-15` existe para pegar.

---

## 9. Emendas necessárias a documentos existentes

Listadas com justificativa. **Nenhuma foi aplicada por este documento.**

### 9.1 `docs/api-contract.md`

| item | emenda | justificativa |
|---|---|---|
| `nbody.initial_conditions` | acrescentar `random_sphere(n, radius, seed, mass_spectrum=None, virial_ratio=0.0, f_cut=0.5, mass_seed=..., vel_seed=..., dtype, device) -> State` | é a IC das Seções 2 e 3; `cold_sphere` permanece intocada e continua sendo a referência bit a bit de `INV-17` |
| novo módulo | `nbody.populations`: amostragem de massas (Seção 2.7) e de velocidades (Seção 3.3), expondo `mass_min_from_mean`, `sample_masses`, `sample_velocities`, `solve_sigma`, e `lambda` como retorno auxiliar | separa a geração estocástica da IC geométrica; sem `lambda` exposto, `INV-16(c)` é intestável |
| novo módulo | `nbody.collisions`: `CollisionModel` (contendo `r_ref`, `v_coh`, `b`, `w`, `frag_f_min`, `eta`, `seed`), `detect`, `pair_disjoint`, `resolve`, e os acumuladores `E_int`, `L_spin` | a colisão é um mapa separável do integrador (Seção 4.5) |
| `integrate()` | parâmetro adicional **somente por palavra-chave** `collision: Optional[CollisionModel] = None` | mantém a assinatura atual válida; `None` é o caminho existente, exigido bit a bit por `INV-30` |
| `integrate()` | levantar `ValueError` quando `collision is not None and softening == 0.0` | `INV-28`; sem isso o campo de aceleração vira `NaN` (Seção 5.2) |
| `nbody.observables` | acrescentar `n_live(state)`, `mass_spectrum_summary(state)`, `scales_from_state(state)` | `N_live` e as escalas realizadas são necessários a `INV-15`, `INV-24` e `INV-31`, e não existem hoje |
| convenção `m = 0` | documentar que slots com `m = 0` são inertes e que `State` os admite; `State.__post_init__` **não** deve rejeitá-los | é o contrato da Seção 5; hoje não está escrito em lugar nenhum |
| `Backend` | nenhuma mudança | o kernel de força é literalmente o mesmo; `m = 0` já é tratado corretamente (verificado, Seção 5.1) |
| contratos de erro | acrescentar que o desfecho de colisão consome exatamente um sorteio por evento aceito, na ordem `(t*, i, j)` | reprodutibilidade (`INV-19(c)`) |

### 9.2 `docs/integradores.md`

| seção | emenda | justificativa |
|---|---|---|
| §2 (nova subseção 2.6) | relação entre softening e raio de contato: com `R_i+R_j < eps` as colisões ocorrem **dentro** da região regularizada, e a velocidade de impacto e o poço de par são os do potencial de Plummer, não os de massas pontuais; com `R_i+R_j > eps` o softening nunca atua e o sistema muda de regime (Seção 4.2 deste documento) | a semântica de `eps` muda qualitativamente quando há contato, e §2.2 não cobre esse caso |
| §7, `INV-10` | `U_MIN_BOUND` deixa de ser `-G m² N(N-1)/(2 eps)` e passa a `-G (M_real² - sum_i m_i²) / (2 eps)` | com massas desiguais a fórmula atual não é a cota; ela subestima ou superestima conforme a variância das massas |
| §7, `INV-9` | a definição operacional de `r_half` passa a ser a mediana de **massa** (Seção 5.3 deste documento); registrar que para `N = 1000` com massas iguais o valor é **bit a bit inalterado** e `COLLAPSE_R_HALF_MIN = 0.3472` continua válido | a fórmula atual mede a mediana errada assim que as massas diferem |
| §7, `INV-3` e `INV-4` | registrar que **não se aplicam** a execuções com fusão ou fragmentação, e nomear os substitutos (`INV-23`, `INV-24`) | sem essa ressalva, um teste correto de `integradores.md` falha contra uma implementação correta com colisões |
| §8.4 ("o que não pode ser afirmado") | acrescentar: nada sobre relaxação de dois corpos ou segregação de massa a partir de execuções com espectro de massas em `3 t_ff` — o tempo de relaxação por segregação é menor que o de relaxação geral por `~m_max/<m> = 285`, mas ainda **[A]** e não medido | com massas desiguais surge a tentação de afirmar segregação, que o horizonte não sustenta |
| §10 | acrescentar item: as extensões estocásticas estão em `docs/simulacao-estocastica.md`; `INV-1..INV-10` permanecem em vigor com as emendas acima | rastreabilidade |

### 9.3 `src/nbody/observables.py`

Linhas `68-74`, `half_mass_radius`. **Confirmado o apontamento.** A implementação atual devolve
`sorted_d[n // 2 - 1]`, a mediana de **contagem**. Substituir pela mediana de **massa** conforme a
Seção 5.3. Compatibilidade bit a bit verificada para massas iguais e `N` par; nenhum valor publicado
muda. **Esta é a única mudança em código existente que este documento exige.**

### 9.4 `docs/glossario.md`

Acrescentar entradas: **razão virial**, **maxwelliana truncada** (com a distinção `f(v)` × `p(|v|)`
da Seção 3.1, que é o ponto mais fácil de enunciar errado em todo o projeto), **raio de contato**,
**detecção varrida**, **energia interna acumulada**, **lei de potência truncada**. A entrada de
amolecimento de Plummer deve ganhar a ressalva da Seção 4.2 sobre colisões dentro da região
regularizada.

---

## 10. Resumo das decisões que este documento fixa

1. **A construção condicionada do espectro de massas é exata, não aproximada** — provada na Seção
   2.6 — **desde que os `k` slots massudos sejam escolhidos por permutação uniforme**. A construção
   enunciada omitia essa condição; ela é agora normativa e testada por `INV-13`, que é o único
   invariante capaz de pegar sua ausência.
2. **`m_min` tem forma fechada**, `m_min = PARTICLE_MASS / g(alpha, R)`, com `R` fixo. Não há raiz a
   buscar. Os casos degenerados `alpha = 1` e `alpha = 2` são obrigatórios e independentes.
3. **O condicionamento em `k ∈ {1,2,3}` descarta `27.8%` da medida** e enviesa a massa média em
   `-0.76%`. O viés é **reportado, não corrigido**; ele entra corretamente nos observáveis pela via
   de `t_ff` ser calculado da massa realizada.
4. **A intuição de que a soma das massas concentra bem está refutada.** A massa é dominada pelo
   extremo inferior (`m^-1.35`), mas a variância é dominada pelo superior (`m^-0.35`), e é a
   variância que decide: `CV(M_tot) = 9.24%`, `sd(t_ff)/t_ff = 4.62%`. Reportar `t_collapse/t_ff`
   com quatro casas sobre uma única semente deixa de ser legítimo.
5. **A ordem das operações de velocidade proposta está correta** — truncar, subtrair a média
   ponderada, reescalar — e o motivo é que o reescalonamento preserva `P = 0` exatamente enquanto a
   subtração muda `K`. Ficam **exatos** `Q` e `P`; fica **aproximado** o teto, que se desloca por
   `lambda = 1 +- 1.3%` e deve ser reportado. Iterar a rejeição é proibido.
6. **`sigma` sai de uma equação implícita no segundo momento truncado**, não de `<v²> = 3 sigma²`.
   Ignorar isso introduz erro **sistemático** de `3.9%`, três vezes o ruído amostral.
7. **`Q = 0.5` com `f_cut = 0.5` é inadmissível e está vetado.** O truncamento impõe
   `Q_sup = 2.009 f_cut² = 0.502`, e o par proposto cai exatamente sobre a singularidade, onde a
   distribuição degenera na bola uniforme e `f(v)` fica **plana** — destruindo a única propriedade
   que justifica a escolha da maxwelliana. Fixados `Q = 0.25`, `f_cut = 0.5`, com verificação em
   tempo de execução de `Q <= 1.532 f_cut²`.
8. **`f(v)` é estritamente decrescente na rapidez; `p(|v|)` não é** — tem pico em `sqrt(2) sigma`.
   O requisito do projeto é satisfeito **exatamente**, na medida `d³v`. A Seção 3.1 fixa a linguagem
   permitida e a proibida. Uma distribuição estritamente decrescente na rapidez foi **rejeitada**:
   ela exigiria `f(v) ∝ |v|^-2`, uma cúspide singular em `v = 0` que não é equilíbrio de
   hamiltoniano algum.
9. **Duas predições falsificáveis sobre `Q`**, ambas baratas: `t_collapse ∝ (1-Q)^(-1/2)` por
   Lagrange–Jacobi (`1.1964` em `Q = 0.25`), e `r_half,min` crescendo linearmente com `Q` por
   barreira de momento angular (`0.4656` em `Q = 0.25`). Medir com massas **iguais**, porque a
   dispersão de ensemble do espectro domina esses efeitos para `Q <~ 0.1`.
10. **Contato acima de `eps` está vetado.** O argumento a favor é formalmente correto e a conclusão
    é errada: `chi = 1` produz `87` colisões por partícula por rebote e coalesce o núcleo em um
    único rebote. Trocar uma imprecisão em `d <~ eps` por uma mudança de regime do sistema inteiro é
    uma troca ruim. Regime **perturbativo** (`chi ≈ 0.1`, `~1` colisão/partícula/rebote) é o padrão,
    com a consequência declarada de que as colisões ocorrem **dentro** da região regularizada.
11. **A álgebra da detecção varrida está correta**, verificada contra minimização por grade. O
    tunelamento **é possível** no regime perturbativo (`C_coll = 1.5` para o par mais rápido a
    `dt = 5e-4`), e é isso que fixa **`DT_COLLISION = 1.25e-4 s`**, quatro vezes menor que
    `DT_COLLAPSE`. `C_coll` medido é invariante obrigatório.
12. **A colisão entra dentro do *drift*, não depois do passo.** É a única forma de aplicar o impulso
    na configuração de contato, e é a paralelidade entre impulso e separação **no ponto de
    aplicação** que faz a colisão elástica conservar `L` exatamente. **"Separar até contato exato"
    está vetado**: injeta energia; no esquema adotado a sobreposição nunca se forma.
13. **O pareamento guloso conserva massa** e é uma **aproximação declarada** da sequência exata de
    eventos, cuja validade é medida por `f_reject <= 0.05`, não presumida. Desempate por `(i, j)` é
    normativo, sem ele não há reprodutibilidade entre dispositivos.
14. **A afirmação sobre a fusão está meio certa.** `ΔK = -(1/2) mu |u|²` **exatamente** —
    confirmado a `1e-15`. Mas `ΔU` **não** é só o termo mútuo: há termos de terceiro corpo que não
    se anulam e **não têm forma fechada**. É essa ausência de forma fechada, e não a fórmula
    alegada, que obriga o acumulador numérico.
15. **A fusão e a fragmentação destroem momento angular** por `-mu (dr x u)` exatamente — o spin do
    par, que o modelo não representa. O enunciado do projeto não mencionava isso. Acumulador
    `L_spin` obrigatório, com `L_total = L_orb + L_spin`; sem ele o efeito atinge `~1e-4` de
    `L_SCALE` e nenhum invariante o veria.
16. **A construção de fragmentação está correta**: massa, momento e `T_cm` conservados exatamente,
    verificados. Ressalva obrigatória: conservar `T_cm` exatamente significa **não dissipar nada** —
    isto não é fragmentação no sentido astrofísico, e é o que permite `E_int` decrescer.
17. **`E_int` é avaliada ATRAVÉS DO MAPA DE DESFECHO, com posições congeladas — nunca ao longo do
    passo.** Avaliá-la sobre o passo absorveria o erro de truncamento do integrador e tornaria
    `E_total` conservada trivialmente, destruindo o diagnóstico que ela existe para salvar. Emenda
    obrigatória à proposta original.
18. **A banda limitada de `|ΔE/E₀|` não sobrevive a fusão ou fragmentação.** Sobrevive: simpletismo
    entre eventos, `P`, e — só para colisões elásticas — `E` e `L` exatos. O substituto é um
    conjunto de quatro peças: `E_total = K+U+E_int`, a curva `E_int(t)`, o resíduo por evento, e
    `L_total`. **A comparação entre integradores continua sendo feita sem colisões.**
19. **Não há crescimento descontrolado com os parâmetros padrão, e há teto fechado:**
    `m*/m_bar = (p_fus + k p_frag)/((1-k) p_frag)` com `k = 3/4 - a/2 = 0.70`, dando `~4.9` (mapa
    padrão) e `5.67` (controle uniforme), isto é `~0.5%` da massa total. **A causa é a fragmentação,
    não o mapa de regime** — o teto existe até no controle uniforme —, e o parâmetro que o controla é
    `a`, não `w`.
20. **O parâmetro de regime `x = T_cm/E_bind` exige o termo gravitacional; omiti-lo está vetado.**
    Sem ele `x >= 1` identicamente por conservação de energia do par, e o canal de fusão fica
    **inalcançável**. Com coesão e gravidade, `x = |u|²/(v_coh² + v_esc_eff²)`, a massa reduzida
    cancela, e a faixa visitada é `x ∈ [0.12, ~75]` — quase três décadas cavalgando `x = 1`, com
    nenhum canal abaixo de `5%`.
21. **O mapa é uma interpolação fenomenológica, não uma teoria de colisões.** Seu conteúdo físico é
    a ordenação e a monotonicidade; `b`, `w` e `v_coh` são calibração. É permitido escrever "sob o
    modelo fixado, `X%` dos eventos foram fusões"; é **proibido** escrever "o colapso produz `X%` de
    fusões".
22. **`m = 0` é exatamente inerte em todos os caminhos verificados** — força, `U`, `K`, `P`, `L`,
    centro de massa —, bit a bit. **A única exceção é `eps = 0` com slot morto coincidente, que
    produz `NaN` via `0 * inf`**; daí a proibição normativa de colisões com `eps = 0`. Slots mortos
    ficam em `r = r_fundido`, `v = v_fundido`.
23. **`half_mass_radius` está errado hoje** e usa a mediana de contagem. Confirmado. A correção para
    mediana de massa é **bit a bit compatível** para massas iguais e `N` par, logo
    `IC_R_HALF_0 = 4.881251` e `COLLAPSE_R_HALF_MIN = 0.3472` permanecem válidos. É a única mudança
    em código existente que este documento exige.
24. **O invariante de ensemble não rejeita o runaway; rejeita a degenerescência.** `K = 32` sementes,
    com critérios sobre `N_final = 1`, número mínimo de eventos, `t_50 > 1 t_ff`, índice de
    dispersão `D = Var(n_merge)/mean(n_merge) <= 5`, e cobertura dos três canais. `D`, e não a
    dispersão de `N_final`, é o instrumento correto: com `~150` fusões esperadas, um critério de
    dispersão sobre `N_final` reprovaria um modelo saudável.
25. **Este documento é anterior à implementação, e a densidade de `[A]` reflete isso.** Todo `[A]`
    nomeia a medição que o converte em `[M]`. Os estágios da Seção 1.3 dizem qual execução produz
    qual número. Nenhum `[A]` pode virar afirmação na prosa do relatório sem passar pela medição que
    ele mesmo declara.
