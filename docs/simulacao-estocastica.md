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

> ## AVISO DE REENQUADRAMENTO — leia antes de qualquer coisa
>
> **Revisão de 2026-08-07.** Este documento foi originalmente escrito sob o critério "o modelo mais
> defensável". O usuário reenquadrou o projeto: **o produto é a VISUALIZAÇÃO**, e aceita-se
> explicitamente **maior erro em troca de menor complexidade**. Onde o rigor custava caro e não
> aparecia na tela, ele foi cortado — **de propósito, não por descuido**.
>
> Quem ler isto daqui a um ano precisa saber que as simplificações desta revisão são deliberadas e
> foram medidas antes de serem adotadas. O registro completo de o quê mudou, por quê, e o que se
> perdeu em cada caso está na **Seção 9.5**. As **oito** coisas que **não** foram simplificadas, e o
> que quebra em cada uma se alguém as simplificar depois, estão na **Seção 4.14 (O PISO)**. A Seção
> 4.14 é a única parte deste documento onde não há liberdade de projeto.
>
> *(Eram nove. O item 6 caiu na revisão (b) de 2026-08-07 — ver 4.14 e 9.6. Este parágrafo dizia
> "nove" até 2026-08-08 (d): sobrevivência da redação original.)*
>
> Regra de leitura: uma aproximação grosseira, uma constante escolhida a dedo ou um modelo
> fenomenológico **não** são defeitos neste projeto. Violar uma conservação exata é.

**Rastreabilidade das afirmações.** Cada afirmação quantitativa está marcada como:
- **[T]** garantido teoricamente (demonstração algébrica dada ou referenciada aqui);
- **[M]** medido nesta configuração específica (valor de referência obtido em fp64);
- **[A]** assumido, ainda não verificado — acompanhado da medição que o decidiria.

Este documento foi escrito **antes** de a implementação existir. Consequentemente há aqui muito menos
**[M]** e muito mais **[A]** do que em `integradores.md`. Isso é deliberado e deve permanecer
visível: todo **[A]** traz explicitamente qual medição o converte em **[M]**. Um **[A]** que vire
número na prosa do relatório sem passar por essa medição é uma falha de processo.

> **Nota de estado, 2026-08-08 (d).** A implementação **existe** desde então (`src/nbody/collisions.py`,
> `src/nbody/populations.py`, `random_sphere`, o parâmetro `collision=` de `integrate`), e três
> execuções colisionais completas foram feitas (Seções 4.13.2, 4.13.4). O verbo no presente
> ("é escrito") era sobrevivência da redação original e está corrigido. **O que NÃO muda:** o
> documento continua vinculante e continua sendo a fonte de onde os testes são escritos **sem ler
> `src/`**. Onde documento e código divergirem, a divergência é relatada — não corrigida de um lado
> nem do outro por conta própria. As divergências conhecidas nesta data estão na Seção 9.1.1.

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

| estágio | o que produz | o que fixa | estado |
|---|---|---|---|
| 1 | espectro de massas + velocidades, sem colisões | `t_ff` realizado, `r_half,min(Q)`, `t_collapse(Q)` | feito |
| 2 | detecção de colisão ligada, sem resolução | taxa de eventos, `chi`, `C_coll` medido, densidade de núcleo | **feito — ver 4.1 e 4.4** |
| 3 | os três desfechos, ensemble de sementes | estatística de canais, `N_final`, `t_50` | **feito.** Execução única duas vezes (4.13.2, 4.13.4) e **ensemble `K_SEEDS = 4` executado (4.13.7)** |

> **Atualização de estado, 2026-08-08 (d).** A linha do estágio 3 dizia "próximo" e já estava
> desatualizada por duas execuções, depois por um ensemble.
>
> # CORREÇÃO 2026-08-08 (d), MESMO DIA — ERRO DE FATO DESTE DOCUMENTO
>
> **Uma versão anterior desta caixa afirmava que o ensemble de `K_SEEDS = 4` "não foi executado".
> Isso estava ERRADO. Ele foi executado, e os resultados estão na nova Seção 4.13.7.** A afirmação
> era minha e não tinha base: eu inferi o estado da campanha do estado do documento, em vez de
> verificar. **O erro é do mesmo tipo que este documento passou a revisão inteira a catalogar** — uma
> premissa não confrontada com a medição — e é registrado aqui em vez de apagado.
>
> **A distinção que eu de fato queria fazer é outra, é válida, e continua valendo mesmo com o
> ensemble executado: variar apenas a semente de COLISÃO, com a mesma condição inicial, não é o
> mesmo que variar a REALIZAÇÃO.** Está enunciada corretamente em 4.13.7, "O que o ensemble testa e
> o que ele não testa".

**O estágio 2 foi executado** (2026-08-07, `chi` grid completa numa única trajetória, 78 s). Os seus
resultados estão incorporados nas Seções 4.1 e 4.4 e converteram em **[M]** três marcas **[A]** que
esta especificação carregava: o valor de `chi`, a densidade numérica do núcleo, e `|u|_max`.
Consequentemente `R_ref` **está fixado por este documento** (Seção 4.1), e não mais pendente de
medição futura.

A calibração do mapa de regime (Seção 4.7) deixou de existir como tarefa: o mapa foi reduzido a uma
forma **sem parâmetro livre** (Seção 4.7 revisada). Não há mais o que calibrar.

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

O expoente `1/3` corresponde a densidade material constante. `chi` é **adimensional**.

> **Revisão 2026-08-07: `chi` está FIXADO em `0.1`, por medição.** A versão anterior desta subseção
> dizia "este documento não fixa `chi`; fixa o que medi-lo significa", e delegava a escolha a uma
> campanha do estágio 2. A campanha foi executada na sua **versão mínima** e o resultado está abaixo.
> Não há mais campanha a rodar, e nenhum implementador precisa tratar `chi` como pendente.

**Grandeza a medir (normativa):** o **número médio de colisões por partícula durante o primeiro
rebote**,

```
N_coll_per_particle := 2 * (numero de eventos detectados em [t_bounce - 0.05 t_ff, t_bounce + 0.05 t_ff])
                       / N_live
```

com `t_bounce = t_collapse`. **Faixa-alvo: `N_coll_per_particle ∈ [0.5, 2]`.** Abaixo de `0.5` as
colisões são invisíveis e a extensão é decoração; acima de `2` a colisão, e não a gravidade, passa a
governar o núcleo (Seção 4.2).

#### 4.1.1 Medição do estágio 2 — a campanha, na sua versão mínima

**Protocolo executado.** Como não há resolução de colisão na linha de base, **a trajetória independe
de `chi`**; logo uma única trajetória serve para toda a grade de `chi` simultaneamente, com o
detector varrido avaliado a cada passo para todos os `R_sum` de uma vez. Isso reduz a campanha de
`5` execuções (uma por `chi`) a **uma**. Configuração: `N = 1000`, massas iguais, `Q = 0`,
`eps = 5.0e-2`, `dt = 5.0e-4`, `12604` passos (`3 t_ff`), fp64, `torch_eager` em GPU, semente de
posições `SEED = 20190222`. Custo total: **77.6 s**.

Um "encontro" é a transição *não sobreposto → sobreposto* do par, e não a detecção repetida a cada
passo: sem resolução, um par que entra em contato **permanece** em contato por muitos passos, e
contar cada passo como evento infla o total em ordens de grandeza. Ver o cabeçalho de
`scripts/collision_rate.py`, que documenta esse viés em detalhe. Este viés existe **somente** na
linha de base sem resolução; com `resolve()` ligado o par se separa por construção e
encontro = detecção.

**Resultado — a estimativa a priori está confirmada** **[M]**:

| `chi` | `R_i+R_j (m)` | encontros em `3 t_ff` | `N_coll_per_particle` **[M]** | a priori **[A]** (tabela abaixo) |
|---|---|---|---|---|
| `0.05` | `0.0050` | `559` | `0.230` | `0.218` |
| **`0.10`** | **`0.0100`** | **`1902`** | **`0.938`** | `0.873` |
| `0.20` | `0.0200` | `7384` | `3.784` | (`0.25` → `5.46`) |
| `0.50` | `0.0500` | `42850` | `22.82` | `21.8` |
| `1.00` | `0.1000` | `159004` | `85.20` | `87.3` |

Concordância dentro de `8%` ao longo de um fator `400` na taxa. **A marca [A] da estimativa a priori
vira [M].**

**DECISÃO NORMATIVA: `chi = 0.1`, `R_ref = 5.0e-3 m = eps/10`.** **[M]** Justificativa: `0.938` cai
no centro da faixa-alvo `[0.5, 2]`. `chi = 0.05` (`0.230`) fica abaixo da faixa e as colisões ficariam
raras demais para aparecer na tela; `chi = 0.2` (`3.78`) já sai da faixa pelo lado colisional.

**Grandezas de rebote medidas na mesma execução** **[M]**:

| grandeza | medido | observação |
|---|---|---|
| `r_half,min` | `0.3457 m` | amostrado **a cada passo** |
| `t_bounce / t_ff` | `1.0354` | contra `1.0361` de `INV-9` |
| partículas dentro de `r_half,min` | `500` | por construção da mediana de massa |
| **densidade numérica do núcleo `n_v`** | **`2888.3 m^-3`** | ver abaixo |
| `\|u\|_max` sobre pares candidatos | `36.3 m/s` | era **[A]** `30 m/s`, Seção 4.4 |

> **Nota sobre `r_half,min = 0.3457` contra o publicado `COLLAPSE_R_HALF_MIN = 0.3472`.** Não há
> conflito: o valor publicado é amostrado a cada `OUT_DT = 1e-2 s`, isto é, a cada `20` passos, e
> esta execução amostrou a **cada passo**, capturando um mínimo `0.4%` mais fundo. O valor publicado
> é um mínimo **subamostrado**, e permanece válido como tal. Nenhum resultado existente muda.

> **A discrepância de densidade de núcleo está RESOLVIDA, e `integradores.md` §4.3 está errado.**
> A versão anterior desta seção registrava a discrepância como **[A]**: `2852 m^-3` do cálculo
> direto contra `~1.4e3 m^-3` declarado em `integradores.md` §4.3, um fator `2`. Medido:
> **`2888.3 m^-3`** **[M]**, dentro de `1.3%` do cálculo direto. **`integradores.md` §4.3 subestima a
> densidade de núcleo por um fator `2` e deve ser corrigido** (emenda na Seção 9.2). Como
> consequência, nenhuma taxa desta seção cai por `2`, e o `chi` recomendado **não** sobe para `0.15`.

#### 4.1.2 A estimativa a priori, preservada para rastreabilidade

Mantida abaixo porque é ela que a medição confirmou, e porque a concordância entre as duas é o que
autoriza confiar no modelo de taxa em regimes não medidos. Núcleo em compressão máxima, `500`
partículas dentro de `r_half,min = 0.3472 m`, massas iguais, `v_rel = sqrt(2) V_CHAR = 4.6386 m/s`,
janela de rebote `0.21 s = 0.1 t_ff` **[A]**:

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
**[T]** Medido, o valor que dá `1` está entre `0.10` (`0.938`) e `0.20` (`3.78`), consistente.

`n_v = 2852 m^-3` era **[A]**; a medição de 4.1.1 deu `2888.3 m^-3` e o converteu em **[M]**.

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

| regime | `chi` | o que acontece (**[M]**, Seção 4.1.1) | estatuto |
|---|---|---|---|
| **perturbativo** | **`0.1`** | `0.94` colisão/partícula/rebote; a colisão é um canal raro sobre o colapso existente; o resultado colisionless permanece o limite `chi -> 0` | **PADRÃO, FIXADO** |
| intermediário | `0.2 – 0.5` | `3.8 – 22.8` colisões/partícula/rebote; o núcleo é colisional; o colapso ainda é reconhecível | sensibilidade |
| dominado | `>= 1` | `85.2`; coalescência; o softening nunca atua | estudo separado |

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

### 4.4 Tunelamento — NÃO existe para o detector varrido, e `dt` NÃO muda

> **REVISÃO NORMATIVA 2026-08-07 — esta seção foi invertida.** A versão anterior concluía
> `DT_COLLISION = 1.25e-4 s = DT_COLLAPSE/4`, quadruplicando o custo de toda campanha colisional e do
> visualizador em modo colisional. **Essa conclusão estava errada, e a origem do erro está
> identificada.** `DT_COLLISION` está **removido**. O passo colisional é
> **`dt = DT_COLLAPSE = 5.0e-4 s`**, o mesmo do colapso suave, e `N_STEPS_COLLISION = 12600`.

#### 4.4.1 De onde veio o erro

O argumento antigo era: com `C_coll > 1` o par atravessa a zona de contato inteira dentro de um
passo, "ao fim do passo os corpos já se separaram, a normal de fim de passo aponta para o lado
errado, e a guarda de aproximação descarta o evento".

**Esse argumento pressupõe que a colisão é resolvida no FIM DO PASSO — exatamente o esquema que a
Seção 4.5 proíbe.** A Seção 4.5 é normativa e insere a colisão **dentro do drift**, em `t*`, com a
normal de contato avaliada em `t*`; ela nunca consulta a configuração de fim de passo e nunca aplica
a guarda de aproximação fora do início do passo. `DT_COLLISION` foi portanto derivado de uma premissa
que o próprio documento revoga duas subseções adiante. **[T]**

#### 4.4.2 Prova: o detector varrido não pode perder um contato, para nenhum `C_coll`

Sobre um passo, o movimento relativo é retilíneo (exato para o drift do Verlet, Seção 4.3), logo

```
d/dt ( dr . dv )  =  |dv|^2  >=  0
```

isto é, **`dr . dv` é monotonicamente não decrescente ao longo do passo**, e a aproximação máxima
ocorre onde `dr . dv = 0`. Consequência imediata: se o instante de aproximação máxima cai no
interior de um passo, então **naquele passo, no seu início, `dr . dv < 0`** — a guarda de aproximação
**passa**, e o `clamp` de `t*` devolve o mínimo interior exato. Se a aproximação máxima cai
exatamente sobre uma fronteira de passo, o passo anterior a detecta com `t* = h`, com a mesma
separação. Em nenhum dos casos há perda. **[T]**

**A colisão frontal é o caso MAIS FÁCIL para o detector varrido, não o mais difícil.** Num encontro
frontal a separação mínima é exatamente `0`, que está tão dentro de `R_i+R_j` quanto é possível
estar; um encontro rasante (`b -> R_i+R_j`) é que é marginal. A intuição de que "uma colisão frontal
exige `C_coll >= 2`" vem de transportar para o varrido o raciocínio correto para um **teste estático
de sobreposição nos extremos do passo** — para esse teste, sim, o par precisa gastar pelo menos um
extremo de passo dentro da zona de contato, o que exige `|u| dt < 2 (R_i+R_j)`. O varrido não faz
esse teste. **[T]**

**Verificação numérica direta** (`4.0e6` encontros sintéticos, geometria exata do detector com a
guarda de aproximação aplicada passo a passo sobre uma grade de fase aleatória) **[M]**:

| caso | `C_coll` varrido | fração perdida |
|---|---|---|
| parâmetro de impacto uniforme no disco de raio `R` | `0.25, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 20.0` | **`0.000000` em todos** |
| frontal (`b = 0.01 R`) | `1.0, 1.5, 1.9, 2.0, 2.1, 3.0, 5.0` | **`0.000000` em todos** |

`200000` realizações por ponto. Nenhuma perda em nenhum regime, incluindo o frontal atravessando
`C_coll = 2`.

#### 4.4.3 Verificação no sistema real: o `dt` menor não compra nada

Mesma trajetória de colapso frio, `chi = 0.1`, `3 t_ff`, os dois `dt` **[M]**:

| `dt` | encontros | `N_coll_per_particle` | `C_coll,max` | conflito de pareamento | wall |
|---|---|---|---|---|---|
| **`5.0e-4`** | **`1902`** | **`0.938`** | **`1.8137`** | **`0.407%`** | **`77.6 s`** |
| `1.25e-4` | `1911` | `0.940` | `0.4535` | `0.462%` | `325.9 s` |

**O quarto do `dt` muda a contagem de colisões em `0.47%` e custa `4.2x` o tempo de parede.** Sob o
critério deste projeto — o produto é a visualização — isso não se paga, e a diferença é invisível.

Note que o conflito de pareamento é ligeiramente **maior** no `dt` menor, não menor: o canal de perda
não é limitado por `dt` nesse regime.

#### 4.4.4 O que se perde, dito com todas as letras

1. **`C_coll,max` passa de `0.45` para `1.81`.** Significa que a configuração de contato em `t*` tem
   a separação mínima resolvida com folga de `~1.8` diâmetros de contato em vez de `~0.45`: o ponto
   da esfera de contato onde o impulso é aplicado é mais arbitrário. **Nenhuma conservação quebra** —
   a reflexão elástica conserva `E`, `P` e `L` exatamente para **qualquer** normal unitária
   (Seção 4.9), e o impulso continua sendo aplicado paralelo à separação real em `t*`.
2. **`~0.5%` dos eventos escorregam um passo**, via adiamento pelo pareamento disjunto.
3. **`C_coll` deixa de ser um critério de validade.** Ele nunca foi um; era um proxy para o
   tunelamento, que não existe. O canal de perda real é o **adiamento pelo pareamento disjunto**, que
   é medido diretamente (Seção 4.5) e vale `0.407%` — doze vezes abaixo do teto de `5%`.

#### 4.4.5 Estatuto novo de `C_coll`: número reportado, não invariante

Definição inalterada, `C_coll = |u| dt / (R_i + R_j)`. **Rebaixado de invariante bloqueante
(`INV-18(b)`, "`<= 1`, sem margem") para grandeza REPORTADA.** A implementação grava
`c_coll_max` no CSV de cada execução e o exibe no HUD; **nenhum teste falha por causa dele**, e
nenhuma execução é invalidada por ele.

Motivo, em uma frase: um número cujo valor medido (`1.81`) e cujo valor quatro vezes menor (`0.45`)
produzem a mesma física dentro de `0.5%` não é uma condição de validade. **[M]**

`|u|_max = 36.3 m/s` **[M]** (era **[A]** `30 m/s`), medido como o máximo de `|v_i - v_j|` sobre
pares candidatos ao longo da execução — exatamente a medição que a versão anterior desta seção
declarava como a que decidiria.

#### 4.4.6 Tabelas de referência, preservadas

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

**Leitura correta desta tabela, após 4.4.2:** ela mostra que `C_coll` **excede `1`** no regime
perturbativo. Ela **não** mostra que colisões são perdidas — a Seção 4.4.2 prova que não são, e a
Seção 4.4.3 mede que não são. A tabela permanece aqui como referência da escala do deslocamento por
passo, que é uma informação útil (`2.3e-3 m` a `4.64 m/s`, contra `R_i+R_j = 0.01 m` em `chi = 0.1`),
não como critério.

Para o par **mais leve** (`R_i+R_j = 2 * 0.1 * eps * 0.6578 = 6.58e-3 m`) e `|u|_max = 36.3 m/s`
**[M]**, `C_coll = 2.76` a `dt = 5.0e-4`. Isso é aceito e reportado.

**DECISÃO NORMATIVA: `dt = DT_COLLAPSE = 5.0e-4 s` para execuções colisionais.
`DT_COLLISION` está REMOVIDO da Seção 8 — não é um símbolo deste projeto.** `N_STEPS_COLLISION`
passa de `50400` a **`12600`** (`3 t_ff`), idêntico a `N_STEPS_COLLAPSE`.

Benefícios operacionais preservados: `dt = 5.0e-4` divide `OUT_DT = 1e-2 s` exatamente (`20` passos
por saída) e é literalmente o passo das escadas de `integradores.md` §8.2, o que torna a comparação
com execuções não colisionais direta em vez de aproximada.

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
**reavaliado**, não simplesmente rejeitado. Rejeitá-lo adia o evento para o passo seguinte, e se por
lá o par já estiver se afastando a guarda de aproximação o descarta de vez. **Este é o ÚNICO canal
pelo qual uma colisão se perde neste modelo** — não há tunelamento (Seção 4.4.2).

**Definição normativa de `f_reject`, POR PASSE.** A versão anterior dizia "sobre toda a execução" sem
definir se o agregado era por passe ou global, e sem dizer como contar quando um contato persiste por
muitos passos — ambiguidade que tornou `INV-19(d)` intestável. Fica assim:

```
f_reject(passe) := (candidatos rejeitados por disjuncao neste passe)
                   / (candidatos detectados neste passe)        ,  e 0.0 se nao ha candidatos

reportado por execucao:   f_reject_max   = max sobre os passes
                          f_reject_total = (soma dos rejeitados) / (soma dos detectados)
```

`AcceptedPairs.f_reject` é **o do passe** (Seção 9.1). Os dois agregados são responsabilidade do
laço de integração, não de `pair_disjoint`. Ambos vão para o CSV e para o HUD.

**Estatuto: REPORTADO, não bloqueante.** A versão anterior fazia `INV-19(d)` exigir
`f_reject <= 0.05` como critério de invalidação de execução. Medido na linha de base do estágio 2 em
`chi = 0.1`: **`f_reject_total = 0.407%`** a `dt = 5.0e-4` e `0.462%` a `dt = 1.25e-4` **[M]** — uma
ordem de grandeza abaixo do teto, e **maior** no `dt` menor, de modo que o teto não é uma condição
sobre `dt`. `0.05` permanece registrado como **linha de atenção**: acima dela, o relatório deve
declarar que a sequência de eventos é uma aproximação grosseira. Nenhum teste falha por causa dele.

> **Ressalva de leitura sobre o `0.407%` medido.** Ele vem da linha de base **sem resolução**, onde um
> par sobreposto permanece candidato por muitos passos consecutivos e contribui muitas instâncias sem
> conflito ao numerador **e** ao denominador — o que **dilui** `f_reject`. O valor medido é uma cota
> provavelmente otimista, não o valor do modelo resolvido. **[A]** — a medição que o decide é
> `f_reject_total` do estágio 3, com `resolve()` ligado, onde o par se separa por construção e a
> diluição desaparece.

**Determinismo.** A chave de ordenação inclui `(i, j)` precisamente para desempatar `t*` idênticos
em ponto flutuante. Sem isso, a ordem de aceitação depende da estabilidade do algoritmo de ordenação
e o resultado deixa de ser reprodutível entre dispositivos. Normativo.

### 4.6 Parâmetro de regime `x`

O desfecho não é sorteado uniformemente: é enviesado por um parâmetro adimensional que compara a
energia do impacto com a energia que mantém o par unido.

> # REVISÃO NORMATIVA 2026-08-07 (b) — O TERMO GRAVITACIONAL ESTÁ RETIRADO
>
> A versão anterior desta seção **vetava** omitir o termo gravitacional de `E_bind`, com três
> argumentos. **O veto está RETRATADO: os três argumentos estavam errados, e o terceiro estava
> exatamente invertido.** O termo gravitacional não regulava o crescimento descontrolado — **ele o
> causava.** A execução do estágio 3 o demonstrou (Seção 4.13.2), e a Seção 4.6.1 mostra por quê.
>
> Consequência: **o item 6 do PISO (Seção 4.14) está retirado**, e `v_esc_eff` deixa de existir.

**Definição normativa:**

```
T_cm   = (1/2) mu |u|^2 ,     mu = m_i m_j / (m_i + m_j) ,   u = v_j - v_i
E_bind = (1/2) mu v_coh^2 ,   v_coh = V_CHAR = sqrt( G M_real / R_0 )

x := T_cm / E_bind  =  |u|^2 / v_coh^2                                             [T]
```

**A massa reduzida cancela, e agora a massa cancela por completo:** `x` depende **apenas da
velocidade relativa no contato**, e de nenhuma propriedade dos corpos. Uma divisão. Nada de `R_sum`,
nada de `eps`, nada de raiz quadrada.

#### 4.6.1 Por que os três argumentos a favor do termo gravitacional estavam errados

**Argumento (1), refutado: "sem ele, `x >= 1` identicamente e a fusão fica inalcançável".**

A derivação supunha que o par chega ao contato **vindo de separação grande**, de modo que
`T_cm(contato) = T_inf + G m_i m_j/sqrt((R_i+R_j)²+eps²)`. **No núcleo colapsado essa premissa é
falsa**, e a razão é o próprio softening: a separação interpartícula local é `~0.0703 m`, comparável
a `eps = 0.05 m`, e o poço de Plummer **já é raso** ali. A queda desde a separação local entrega só
uma fração do poço **[T]**:

| par | profundidade no contato / na separação local | fração de `v_esc_eff²` ganha | piso real de `x` |
|---|---|---|---|
| `m_bar`–`m_bar` | `1.692` | `40.9%` | **`0.409`**, não `1.0` |
| `300 m_bar`–`m_bar` | `1.367` | `26.9%` | **`0.269`**, não `1.0` |

Ou seja, **mesmo com `E_bind` puramente gravitacional o piso de `x` nunca foi `1`** — era `~0.4`, e
cai ainda mais para corpos massudos. O argumento que originou o veto estava errado **desde a
primeira versão do documento**, e a medição apenas o expôs.

**Argumento (2), verdadeiro mas irrelevante: "o termo não é desprezível numericamente".** Ele é de
fato grande. Ser grande é precisamente o problema, não a justificativa — ver (3).

**Argumento (3), INVERTIDO: "ele é o mecanismo que regula o crescimento descontrolado".**

É o oposto. `v_esc_eff² = 2GM/sqrt(R_sum²+eps²)` cresce **quase linearmente com `M`** no regime
`chi = 0.1`, porque `R_sum ∝ m^(1/3)` permanece bem abaixo de `eps` até `m ~ 1000 m_bar`. O
numerador `|u|²` vem da dinâmica de N corpos e permanece na escala do núcleo. Logo `x -> 0` conforme
o corpo cresce, e `p_fus -> 1`: **realimentação positiva pura.** Medido em campo médio, para um corpo
de massa `m` colidindo com um `m_bar` a `|u| = 4.639 m/s` **[T]**:

| `m/m_bar` | `v_esc_eff²` | `x` | `p_fus` | `p_frag` |
|---|---|---|---|---|
| `1` | `5.24` | `1.345` | `0.146` | `0.264` |
| `7` | `20.5` | `0.688` | `0.283` | `0.134` |
| `22` | `57.4` | `0.316` | `0.489` | `0.049` |
| `50` | `123.3` | `0.161` | `0.663` | `0.017` |
| `300` | `636.9` | `0.033` | `0.908` | `0.001` |

**`p_fus` cruza `1/2` em `m ≈ 22 m_bar`** (exatamente em `x = (sqrt13 - 3)/2 = 0.3028`) **[T]**.
Acima disso o corpo funde mais vezes do que não funde, e o canal de fragmentação — que era o
mecanismo de contenção — **se apaga justamente quando seria necessário**.

**Nota estrutural, que continua verdadeira e agora tem o sinal certo:** é o **softening**, não o raio
de contato, que fixa a profundidade do poço no regime perturbativo (`R_sum << eps`). Era por isso que
`v_esc_eff²` seguia `2GM/eps` de forma tão limpa — e é por isso que a dependência com `M` era tão
pura. A propriedade estava documentada; a sua consequência é que não estava.

**Custo da retirada.** `p_fus` deixa de depender da massa. Nada mais muda: os três canais continuam
alcançáveis (Seção 4.8 revisada), e o piso de `x` desaparece por completo em vez de existir em
`~0.4`.

**Adimensionalização de `v_coh` — sem parâmetro.**

```
v_coh = V_CHAR = sqrt( G M_real / R_0 )                  recalculado da massa realizada (Sec. 2.10)
```

**`COH_VELOCITY_FACTOR` está REMOVIDO da Seção 8: `v_coh` É `V_CHAR`, sem fator.** A versão anterior
o expunha como constante ajustável com valor padrão `1.0`; sob o critério desta revisão, um parâmetro
cujo único valor justificado é `1.0` é um parâmetro a menos. `V_CHAR` é a única escala de velocidade
que o problema já tem, já é computada por `scales_from_state`, e não custa nada.

**A energia de coesão é agora a ÚNICA escala de `E_bind`, e é a forma mais simples possível dela.**
Com o termo gravitacional retirado, `v_coh` sozinho fixa a escala, e ele não tem parâmetro.

#### 4.6.2 A alavanca pré-declarada NÃO resolvia isto — e por que a troca não é ajuste post-hoc

Este documento declarou, **antes de qualquer resultado**, que a única alavanca autorizada seria
elevar `v_coh` acima de `V_CHAR`. É necessário dizer com franqueza o que ela faz e o que não faz.

**Ela não resolve o crescimento descontrolado, e demonstra-se que nenhum valor dela resolveria.**
`p_fus > 1/2` exige `x < 0.3028`, isto é `|u|² < 0.3028 (v_coh² + v_esc_eff²)`. Com
`v_esc_eff² ∝ M` sem cota superior e `|u|` limitado pela dinâmica do núcleo, a desigualdade passa a
valer para `M` grande o bastante **qualquer que seja `v_coh`**. **[T]** A alavanca move a massa em
que `p_fus` cruza `1/2`; não elimina a travessia. Pior: elevá-la **aumenta** `p_fus` em toda parte,
que é a direção errada para este defeito.

**Por que ela foi declarada, e por que continua válida no seu escopo.** Ela foi declarada para uma
falha **diferente e específica**: `INV-31(C5)` reprovando por **falta** de fusão (canal abaixo de
`5%`). Para essa falha ela é a alavanca certa, e **permanece em vigor para ela**. A falha observada
foi a oposta — fusão em excesso, dependente da massa — e nunca esteve no escopo declarado.

**Por que trocar de alavanca aqui não viola a regra deste documento.** A regra é que a resposta a uma
falha seja justificada **pelo mecanismo**, não pelo número, e que a correção tivesse sido a escolha
certa caso o mecanismo fosse conhecido antes. As duas condições se verificam:

- A causa identificada é a **dependência de `x` com a massa**, via `v_esc_eff`. A correção adotada —
  retirar `v_esc_eff` — é exatamente a remoção dessa dependência, e não um ajuste de nível.
- Se soubéssemos, ao escrever a Seção 4.6, que o par chega ao contato com apenas `~40%` da energia
  de queda desde o infinito (Seção 4.6.1), **o termo nunca teria sido posto lá**: ele foi introduzido
  para corrigir um piso `x >= 1` que nunca existiu. A correção não é uma alavanca nova; é a
  **retirada de um erro**.
- A correção **reduz** complexidade: apaga `v_esc_eff`, uma raiz quadrada e uma soma por evento, e
  toda a dependência de `E_bind` com `R_sum` e `eps`. Um ajuste post-hoc típico acrescenta um
  parâmetro; este remove um termo.

**Alavanca em vigor a partir de agora, declarada antes do próximo resultado:** se `INV-31(C5)`
reprovar **por falta** de fusão, elevar `v_coh` acima de `V_CHAR`. Se reprovar **por excesso**,
reduzir `v_coh` abaixo de `V_CHAR`. Em ambos os casos `x = |u|²/v_coh²` desloca-se uniformemente e
**sem** reintroduzir dependência com a massa. Nenhuma outra alavanca é autorizada.

### 4.7 O mapa `x -> (p_fus, p_el, p_frag)` — sem parâmetro livre

> **REVISÃO NORMATIVA 2026-08-07.** A versão anterior usava um softmax sobre escores lineares em
> `ln x`, com dois parâmetros de forma (`b`, `w`), um `clamp` em `±30 w`, avaliação obrigatória por
> *log-sum-exp*, e três constantes de configuração (`MAP_B`, `MAP_W`, `MAP_S_CLAMP`) marcadas **[A]**
> pendentes de calibração. **Está substituído pela forma abaixo, que é o mesmo mapa em `w = 1`,
> `b = ln 3`, escrito sem transcendental nenhuma e sem parâmetro algum a calibrar.**
> `MAP_B`, `MAP_W` e `MAP_S_CLAMP` estão **removidos da Seção 8**.

**Definição normativa.**

```
x = clamp( x , 1/X_CLAMP , X_CLAMP )              X_CLAMP = 1.0e12

Z = 1/x + 3 + x

p_fus  = (1/x) / Z
p_el   =    3  / Z
p_frag =    x  / Z
```

Três divisões e duas somas. Sem `exp`, sem `log`, sem *log-sum-exp*, sem subtração de máximo.

**Propriedades — todas [T], e todas as que a versão anterior tinha:**

- **Soma exatamente `1`** por construção (numeradores somam `Z`).
- **Estritamente positivas para todo `x`.** Nenhum canal é jamais proibido.
- **Monotonicidade estrita**, que é todo o conteúdo físico do mapa: `p_fus` estritamente decrescente
  em `x`, `p_frag` estritamente crescente. Lento funde, rápido fragmenta.
- **`p_el` máxima em `x = 1`**, porque `Z = 3 + (x + 1/x)` é mínima em `x = 1` (`AM-GM`); vale
  `p_el(1) = 3/5 = 0.6` exatamente.
- **Simetria exata sob `x -> 1/x`**: troca `p_fus` e `p_frag` e preserva `p_el`. **[T]** Esta
  propriedade é nova — o softmax só a tinha aproximadamente — e é um teste barato e forte.
- **Travessias em forma fechada**: `p_fus = p_el` em `x = 1/3`; `p_frag = p_el` em `x = 3`. O platô
  elástico é **exatamente uma década, centrada em `x = 1`**.

**Estabilidade numérica (normativa).** O `clamp` em `X_CLAMP = 1e12` existe para honrar literalmente
o requisito "nenhuma probabilidade exatamente zero", e a sua derivação é: nos extremos, a menor
probabilidade vale `(1/X_CLAMP)/(X_CLAMP + 3) ≈ X_CLAMP^-2 = 1e-24`. **[T]** Isso está acima do menor
normal de **fp32** (`1.18e-38`) com `14` ordens de folga, e `Z <= ~1e12` está abaixo do maior fp32
(`3.4e38`) com `26` ordens de folga. **O mapa é seguro em fp32 e fp64 sem nenhum cuidado adicional.**

> Por que `1e12` e não `1e30`: com `X_CLAMP = 1e30` a menor probabilidade seria `1e-60`, que
> **subnormaliza a `0.0` em fp32** e violaria o requisito. `1e12` é a maior potência redonda de `10`
> que mantém `X_CLAMP^-2` seguro em fp32. A faixa visitada é `x ∈ [0.12, 108]` (Seção 4.8), de modo
> que o `clamp` **nunca atua na prática** e existe só como guarda.

**Valores** **[T]**, aritmética direta:

| `x` | `p_fus` | `p_el` | `p_frag` |
|---|---|---|---|
| `0.01` | `0.97078` | `0.02913` | `0.00010` |
| `0.1` | `0.76336` | `0.22901` | `0.00763` |
| `0.123` (piso visitado) | `0.73045` | `0.26954` | `0.01105` |
| `1/3` (travessia) | `0.47368` | `0.47368` | `0.05263` |
| `1` | `0.20000` | `0.60000` | `0.20000` |
| `1.67` (típico do núcleo) | `0.11365` | `0.56939` | `0.31696` |
| `3` (travessia) | `0.05263` | `0.47368` | `0.47368` |
| `6.4` | `0.01635` | `0.31393` | `0.66972` |
| `20` | `0.00217` | `0.13015` | `0.86768` |
| `108` (teto visitado) | `0.00008` | `0.02712` | `0.97279` |

**Caso de controle uniforme.** O softmax tinha `w = inf` como entrada válida devolvendo
`(1/3, 1/3, 1/3)`. Esta forma não tem `w`. Onde o controle uniforme for necessário — e ele é, na
Seção 4.12 — ele é obtido **substituindo o mapa por `(1/3, 1/3, 1/3)` constante**, não por um valor
limite de parâmetro. A implementação expõe isso como uma escolha explícita, não como um `w` grande.

**O que se perde nesta simplificação.** `p_fus` no núcleo cai de `0.242` (softmax `w = 3`) para
`0.114`: o mapa é **mais duro**, e haverá aproximadamente **metade** das fusões. Isso é aceito, e é
até favorável ao critério de não degenerescência da Seção 4.13 — menos coalescência, menos risco de
terminar em poucos corpos. O custo é que o canal de fusão fica mais próximo do piso de `5%` de
`INV-31(C5)`; a alavanca declarada, se ele reprovar, é `v_coh` (Seção 4.6), **não** o mapa.

#### 4.7.1 Sorteio — exatamente 2 uniformes por evento aceito, sempre

> **Esta subseção fecha uma lacuna real da versão anterior.** Ela dizia "um único uniforme por evento
> aceito", e a Seção 4.9 então precisava de mais sorteios (`f`, e uma direção isotrópica cujo método
> de amostragem — e portanto cuja contagem de sorteios — nunca foi especificado). Com isso, `INV-19(c)`
> exigia determinismo bit a bit sobre um fluxo cujo consumo não estava definido: **inatingível**, e
> uma fonte garantida de divergência entre implementação e teste.

**Normativo, sem ambiguidade:**

```
rng_c = numpy.random.default_rng(COLLISION_SEED)     # quarto fluxo, separado (Sec. 8)

para cada evento aceito, na ordem do array AcceptedPairs
(que a Secao 4.5 garante ordenado por (t*, i, j)):

    u1 = rng_c.random()          # 1o sorteio: SEMPRE consumido
    u2 = rng_c.random()          # 2o sorteio: SEMPRE consumido, mesmo que nao seja usado

    if   u1 <  p_fus            ->  fusao          (u2 descartado)
    elif u1 <  p_fus + p_el     ->  elastica       (u2 descartado)
    else                        ->  fragmentacao   (u2 fornece f, Secao 4.9)
```

**Os dois pontos que tornam isto testável, e que não podem ser trocados por nada equivalente:**

1. **`u2` é sorteado INCONDICIONALMENTE, antes do desvio de canal.** Sortear `u2` só dentro do ramo
   de fragmentação daria um consumo de fluxo dependente do canal, e portanto uma trajetória de
   números aleatórios que muda quando a física muda — o oposto de reprodutível. **O passo é fixo em
   `2` sorteios por evento aceito, para todo canal.**
2. **Não há terceiro sorteio.** A direção isotrópica da fragmentação foi **eliminada** (Seção 4.9): a
   velocidade relativa de saída é alinhada com a normal de contato, que já está calculada.

Consequência direta e testável (`INV-32`): após um passe com `n_events` eventos aceitos, o
`Generator` consumiu **exatamente `2 * n_events`** valores. Um passe com zero eventos consome zero.

### 4.8 Faixa de `x` que a simulação realmente visita

> **REESCRITA em 2026-08-07 (b).** A tabela anterior decompunha `x` em função de `|u_inf|`, supondo
> **queda isolada de dois corpos desde o infinito**. A Seção 4.6.1 mostra que essa não é a
> configuração que a dinâmica entrega, e a execução do estágio 3 confirmou. A decomposição correta,
> com `v_esc_eff` retirado, é diretamente em `|u|` — a velocidade relativa **real no contato**, que
> é o que a detecção mede e reporta em `rel_speed`.

Esta é a pergunta que decide se o mapa é física ou enfeite. Com `v_coh = V_CHAR = 3.2800 m/s` e
`x = |u|²/v_coh²`, sobre a faixa de `|u|` medida no estágio 2 (`|u|_max = 36.3 m/s` **[M]**):

| `\|u\|` (m/s) | `x` | `p_fus` | `p_el` | `p_frag` | quem é |
|---|---|---|---|---|---|
| `0.5` | `0.0232` | `0.9344` | `0.0651` | `0.0005` | vizinhos quase comoventes |
| `1.0` | `0.0930` | `0.7767` | `0.2166` | `0.0067` | encontro lento |
| `2.0` | `0.3718` | `0.4437` | `0.4949` | `0.0613` | travessia fusão/elástica |
| `3.28` | `1.0000` | `0.2000` | `0.6000` | `0.2000` | `\|u\| = v_coh`, pico elástico |
| `4.64` | `2.0012` | `0.0908` | `0.5454` | `0.3638` | **típico do núcleo** |
| `8.0` | `5.9489` | `0.0184` | `0.3291` | `0.6525` | encontro rápido |
| `15.0` | `20.914` | `0.0020` | `0.1252` | `0.8728` | rebote |
| `36.3` | `122.48` | `0.0001` | `0.0239` | `0.9760` | par mais rápido medido |

**Conclusão: a faixa visitada é `x ∈ [~0.02, ~122]`, quase quatro décadas, cavalgando `x = 1`.**
Nenhum canal é proibido, e — o ponto que importa — **`x` não depende mais de massa alguma**, logo
nenhum corpo pode empurrar a si próprio para dentro de um canal.

**A ressalva anterior sobre corpos massudos está RETIRADA.** Ela dizia que corpos massudos teriam
`x` preso perto de `1` e que isso os "auto-regulava". Ambas as metades estavam erradas: `x` não
ficava preso perto de `1`, ia a `0.03`; e o efeito não regulava, realimentava. Ver 4.6.1 e 4.13.2.

**Risco declarado da nova forma.** Com `x = |u|²/v_coh²`, a fusão depende inteiramente de existirem
encontros **lentos** (`|u| < ~2 m/s`). Se a distribuição de `|u|` no núcleo for mais dura que isso, o
canal de fusão pode ficar abaixo dos `5%` de `INV-31(C5)`. **[A]** — a medição que decide é a
estatística de canais do próximo estágio 3, e a alavanca declarada para essa falha é `v_coh`
(Seção 4.6.2).

**Estatuto epistêmico do mapa, dito sem rodeios.** O mapa é uma **interpolação fenomenológica**, não
uma teoria de colisões. Seu conteúdo físico é a **ordenação** (fusão em `x` baixo, fragmentação em
`x` alto, elástica no meio) e a monotonicidade; a constante `3` e a escala `v_coh` são convenção.
Consequência normativa para o relatório: **as frações de canal realizadas são uma saída da
parametrização, jamais uma predição física.** É permitido escrever "sob o modelo fixado na Seção 4.7,
`X%` dos eventos foram fusões"; é proibido escrever "o colapso produz `X%` de fusões".

### 4.9 Os três desfechos

Notação comum: `M = m_i + m_j`, `mu = m_i m_j / M`, `u = v_j - v_i`, `V = (m_i v_i + m_j v_j)/M`,
`T_cm = (1/2) mu |u|²`. Tudo avaliado na configuração de contato, em `t*`.

#### (0) Regra de slots — normativa, e o índice menor sempre vence

> **Esta subseção fecha uma lacuna real da versão anterior**, que dizia "slot liberado: `m = 0`" sem
> dizer **qual** dos dois slots é liberado, e descrevia os fragmentos como `a` e `b` sem dizer qual
> vai para qual slot. Sem isto, implementação e teste divergem silenciosamente, e `INV-19(c)`
> (determinismo bit a bit) é inatingível.

Para todo evento aceito `(i, j)`, a Seção 4.5 garante `i < j`. **O índice menor sempre fica com o
corpo (ou com o primeiro fragmento).** Sem exceção, para os três canais:

| canal | slot `i` recebe | slot `j` recebe |
|---|---|---|
| **elástica** | `m_i`, `r_i`, `v_i'` | `m_j`, `r_j`, `v_j'` |
| **fusão** | `m = M`, `r = r_c`, `v = V` | `m = 0`, `r = r_c`, `v = V` (slot morto, Seção 5) |
| **fragmentação** | fragmento `a`: `m_a = f M`, `r_a`, `v_a` | fragmento `b`: `m_b = M - m_a`, `r_b`, `v_b` |

Na fusão, o slot morto recebe **a posição e a velocidade do corpo fundido**, `r_j = r_i = r_c` e
`v_j = v_i = V`, conforme a Seção 5.2, ponto 2. Não é permitido parqueá-lo em outro lugar nem zerar
`r_j`.

#### (0.1) A normal de contato, e o caso degenerado `|sep| = 0`

A normal `n = sep / |sep|`, com `sep = dr + t* dv`, é usada pelos três canais: direção do impulso na
elástica, direção de saída na fragmentação, e eixo de colocação dos fragmentos. Quando `|sep| = 0`
exatamente — colisão frontal com parâmetro de impacto nulo — ela é `0/0`.

**REGRA NORMATIVA:**

```
sep     = dr + t* dv
sep_sq  = sep . sep

se sep_sq == 0.0 :   n = -u / |u|          (u = v_j - v_i, avaliado em t*)
senao            :   n =  sep / sqrt(sep_sq)

se |u| == 0 tambem : fora de contrato -> ValueError nomeando o par (i, j)
```

**Isto não é um fallback arbitrário: é o limite da própria normal.** Perto de `t*` o movimento
relativo é retilíneo, logo `sep(t) = (t - t*) u`. Para `t < t*` (aproximando-se, que é a condição da
guarda da Seção 4.3), `sep(t) = -(t* - t) u`, portanto

```
lim_{t -> t*^-}  sep(t) / |sep(t)|  =  - u / |u|                                   [T]
```

O limite existe, é único, e a regra é a **extensão contínua** da definição — não uma escolha de
conveniência. Verificações de sinal e de consequência, todas **[T]**:

- `n` aponta de `i` para `j`, como exige a convenção de 4.9.0, porque `dr` encolhia ao longo de `-u`;
- `u . n = -|u| < 0`, isto é, o par continua "aproximando-se" no sentido da guarda — a semântica não
  se inverte no limite;
- **elástica:** `u' = u - 2(u.n)n = u - 2u = -u`. Reversão frontal exata, que é precisamente o
  desfecho físico correto de um choque de cheio;
- **fragmentação:** `u' = |u| sqrt(mu/mu') * (-u/|u|)` — os fragmentos recuam ao longo do eixo de
  chegada, e a colocação em `r_c ± (·)(R_a+R_b) n` os separa normalmente.

**Por que a regra não toca nenhum item do PISO (Seção 4.14).** O item 1 é o único que
poderia estar em risco, e ele é **imune neste caso específico**: a destruição de `L` pelo impulso é
`ΔL = -(dr) x J`, avaliada no ponto de aplicação, e ali `dr = sep = 0` **exatamente**. Logo
`ΔL = 0` para **qualquer** `J`, portanto para qualquer `n` unitária. `K` conserva por reflexão (para
qualquer `n` unitária), `P` conserva por impulsos opostos, e a massa não é tocada. Os itens 2–5 e 9
não envolvem `n`. O item 7 usa `d_ij = |sep| = 0`, e `E_grav(m, m', 0) = G m m' / eps` é
**finita** — garantida pelo item 8, `eps > 0`. **Nenhum item do PISO é atingido, e o item 8 é o que
mantém o caso degenerado inofensivo rio abaixo.**

**Sobre `dr = 0` e `dv = 0` simultâneos: é inalcançável, e por isso levanta exceção.** A guarda da
Seção 4.3 é `dr . dv < 0` **estrita**; com `dv = 0` tem-se `dr . dv = 0`, que não satisfaz a guarda,
de modo que **`detect` nunca emite um candidato com `|u| = 0`**. Um par assim só pode chegar a
`resolve` por um chamador que ignorou `detect` — o que os testes fazem, ao construir pares
sintéticos. `ValueError` é a resposta certa: informa ao autor do teste que a entrada está fora de
contrato, e não pode disparar numa execução real. **Não** inventar uma direção arbitrária aqui: sem
`dr` e sem `u`, não existe direção alguma no problema, e qualquer escolha seria física fabricada.

**Não há limiar, e um limiar seria pior que inútil.** A condição do ramo é exatamente
`sep_sq == 0.0` como computado, e nada mais. Duas razões, ambas medidas **[M]**:

1. **Não existe faixa em que a normal verdadeira seja computável mas ruim.** `sep/|sep|` tem erro
   relativo de poucos ulp em **qualquer** magnitude não nula; não há degradação ao aproximar-se de
   zero. `sep_sq` só subnormaliza a `0.0` abaixo de `|sep| ≈ 2.2e-162` em fp64 e `5.3e-23` em fp32 —
   isto é, `160` e `21` ordens de grandeza abaixo de `R_i+R_j = 1e-2 m`. O ramo degenerado é
   alcançado por **cancelamento exato**, jamais por underflow de uma separação fisicamente
   significativa.
2. **Um limiar substituiria uma normal boa pelo fallback**, introduzindo erro onde não havia, e
   acrescentaria uma constante para se discutir. É complexidade que compra prejuízo.

**Frequência — e ela explica quanto isto vale.** **[M]**, `2e5` realizações por caso:

| configuração | `sep_sq == 0.0` |
|---|---|
| encontros genéricos (a execução real) | **`0 / 200000`**; menor `\|sep\|` visto: `2.0e-4 m` |
| frontal colinear sobre um eixo coordenado | **`65.3%`** |
| frontal colinear sobre um eixo genérico | **`19.6%`** |

Leitura: em fp64 o caso tem **medida nula numa execução real** — exige `dr` exatamente paralelo a
`dv` — e **nunca aconteceu** em `2e5` encontros genéricos. Mas é **quase certo numa suíte de
testes**, que constrói configurações simétricas e colineares de propósito — o mesmo padrão que a
Seção 4.3 já registra para `|dv|² = 0`. **A regra existe para os testes, não para a simulação**, e é
por isso que ela vale exatamente um ramo `if` e nem uma linha a mais.

**Testável** dentro dos procedimentos já existentes de `INV-20` (elástica) e `INV-22`
(fragmentação), acrescentando aos seus conjuntos de eventos sintéticos ao menos um par frontal
colinear com `sep_sq == 0.0` verificado, e exigindo: ausência de `NaN` em `r`, `v`, `m`; `u' = -u`
dentro de `100 eps_prec` na elástica; e as conservações de `m`, `P`, `L`, `K` já tabeladas — que
neste caso valem com `ΔL = 0` **exato**, e não apenas dentro de tolerância.

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
r_c    = ( m_i r_i + m_j r_j ) / M          (centro de massa do par, em t*)
V      = ( m_i v_i + m_j v_j ) / M

slot i:   m_i <- M       r_i <- r_c     v_i <- V        (corpo fundido; indice MENOR, Sec. 4.9.0)
slot j:   m_j <- 0       r_j <- r_c     v_j <- V        (slot morto, Sec. 5)
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

> **REVISÃO NORMATIVA 2026-08-07: a direção isotrópica sorteada foi ELIMINADA.** A versão anterior
> mandava sortear uma direção isotrópica `n_iso` para a velocidade relativa de saída, sem especificar
> o método de amostragem — e portanto sem fixar quantos sorteios um evento de fragmentação consome,
> o que tornava `INV-19(c)` inatingível (Seção 4.7.1). **A velocidade relativa de saída é agora
> alinhada com a normal de contato `n`, que já está calculada.**

```
n      = ( dr + t* dv ) / | dr + t* dv |    (normal de contato em t*, aponta de i para j)

f      = 0.1 + 0.8 * u2                     (u2 e o 2o uniforme do evento, Sec. 4.7.1)
m_a    = f * M
m_b    = M - m_a                            <-- NAO (1-f)*M ; ver nota de ponto flutuante
mu'    = m_a m_b / M
u'     = |u| * sqrt(mu/mu') * n              <-- ao longo da normal, separando-se

slot i (fragmento a):   m_a ,  r_a = r_c + (m_b/M)(R_a+R_b) n ,  v_a = V + (m_b/M) u'
slot j (fragmento b):   m_b ,  r_b = r_c - (m_a/M)(R_a+R_b) n ,  v_b = V - (m_a/M) u'
```

`R_a` e `R_b` são os raios de contato das massas **novas** (`R = R_ref (m/m_bar)^(1/3)`, Seção 4.1),
não os antigos.

**Por que isto é seguro, e por que é melhor.** Nenhuma das provas de conservação abaixo usa a
**direção** de `u'` — todas usam apenas `|u'|` e a antissimetria da colocação. Logo massa, `P`,
`T_cm`, `K` e `sum_i m_i r_i` continuam exatamente conservados, sem alteração. **[T]** Além disso:

- `n` aponta de `i` para `j` e o par está se aproximando em `t*` (`u . n < 0`), logo `u'` com sinal
  `+n` é **separação**: os fragmentos saem afastando-se, o que reforça a guarda de aproximação e
  impede recolisão imediata, exatamente como a construção antiga pretendia;
- fragmentos saindo ao longo do eixo do impacto é **fisicamente mais razoável** que isotrópico;
- elimina `3` sorteios por evento e fixa o consumo do fluxo em `2` (Seção 4.7.1);
- torna a fragmentação a versão do choque elástico com redistribuição de massa: mesma direção, mesma
  `T_cm`, partição de massa diferente.

**O que se perde:** a cláusula de isotropia de `INV-22` deixa de ter objeto e foi **removida**
(Seção 6, `INV-22`). Nenhuma outra cláusula de `INV-22` muda.

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
(`|r_a - r_b| = R_a + R_b`, com os novos raios) e **ao longo de `n`** (que agora é também a direção de
`u'`), portanto separando-se. Isso:
(i) preserva `sum_i m_i r_i` exatamente; (ii) impede recolisão imediata, reforçando a guarda de
aproximação; (iii) altera `U` — o termo mútuo passa de `-G m_i m_j/sqrt(d_ij²+eps²)` para
`-G m_a m_b/sqrt((R_a+R_b)²+eps²)`, **em forma fechada**, e os termos de terceiro corpo mudam por um
resíduo pequeno e medido. Ver a Seção 4.10 revisada: **só o termo mútuo entra em `E_int`**, e o
resíduo de terceiros é declarado, não computado.

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
(`INV-31(C6)`). Se o moinho for excessivo, a mitigação declarada é multiplicar o numerador de
`p_frag` por um fator contínuo `Lambda(M) ∈ (0, 1]` que decresce com `M` — isto é,
`(1/x, 3, Lambda(M) * x) / Z` com `Z` renormalizado —, o que preserva a positividade estrita.
**Um corte duro é proibido**, porque violaria o requisito de que nenhum canal tenha probabilidade
exatamente zero. **[A]** — a medição que decide se isso é necessário é `INV-31(C6)`.

### 4.10 Os acumuladores `E_int` e `L_spin`

**A proposta do enunciado está certa na forma e ERRADA no ponto de avaliação. Emenda obrigatória.**

O enunciado propõe `E_int += E_mec(antes) - E_mec(depois)` "avaliado sobre o passe inteiro". Se
"passe inteiro" significar o passo de integração, a contabilidade absorve também o **erro de
truncamento do integrador**, e `E_total = K + U + E_int` passa a ser conservada trivialmente,
medindo exatamente nada. O diagnóstico-chefe do projeto seria destruído pela própria máquina
construída para salvá-lo.

**Definição normativa: os acumuladores são avaliados ATRAVÉS DO MAPA DE DESFECHO, com as posições
congeladas em `t*`, e nunca ao longo do passo.** Isso é inalterado e continua sendo o ponto que
impede a contabilidade de absorver o erro de truncamento do integrador.

> **REVISÃO NORMATIVA 2026-08-07: `E_int` é computada em FORMA FECHADA ao nível do par, `O(1)` por
> evento.** A versão anterior exigia recomputar `ΔU` com a soma completa de terceiros, `O(N)` por
> evento, e `INV-23(a)` exigia concordância ao nível da precisão de máquina. **O termo de terceiros
> está removido do cálculo e passa a ser um resíduo DECLARADO E MEDIDO**, não computado.

> # CORREÇÃO DE SINAL 2026-08-07 (b) — ERRO DESTE DOCUMENTO, NÃO DA IMPLEMENTAÇÃO
>
> A expressão de **fragmentação** publicada na revisão anterior estava com **o sinal invertido**:
> dizia `E_int += E_grav(antes) - E_grav(depois)`. A derivação correta a partir da própria regra
> geral do documento, `E_int += -(ΔK + ΔU)`, dá o oposto. Derivação, para não restar dúvida:
>
> ```
> U_mutuo = -E_grav          (E_grav e' definida POSITIVA)
> fragmentacao:  dK = 0 ,  dU = (-E_grav_depois) - (-E_grav_antes) = E_grav_antes - E_grav_depois
> E_int += -(dK + dU) = E_grav_depois - E_grav_antes                                 [T]
> ```
>
> **Uma implementação fiel ao documento reproduz o erro**, e reproduziu: é a causa direta do
> `E_int/|E_0| = -109.8` observado (Seção 4.13.2). A expressão de **fusão** estava e permanece
> correta.

**Definição normativa revisada e CORRIGIDA — três expressões, todas `O(1)`:**

```
para cada evento aceito, com as posicoes congeladas em t*, e com
    d_ij = | dr + t* dv |          (separacao no contato)
    E_grav(m,m',d) := G m m' / sqrt( d^2 + eps^2 )        (POSITIVA; vale |U_mutuo|)

    elastica:        E_int += 0
    fusao:           E_int += T_cm - E_grav(m_i, m_j, d_ij)
    fragmentacao:    E_int += E_grav(m_a, m_b, R_a + R_b) - E_grav(m_i, m_j, d_ij)   <-- SINAL CORRIGIDO

    L_spin += mu ( dr(t*) x u )      para fusao e fragmentacao
    L_spin += 0                      para elastica
```

**Verificação de sanidade, obrigatória de reproduzir num teste.** Se a fragmentação aprofunda o poço
mútuo (`E_grav_depois > E_grav_antes`), então `U` cai, `K` não muda, logo `E_mec = K + U` **cai** —
energia saiu do orçamento mecânico, o que é **dissipação**, e `E_int` tem de **subir**. A expressão
corrigida sobe. A anterior descia. Este raciocínio de uma linha é o teste que pega a inversão.

`T_cm = (1/2) mu |u|²`. Todas as quantidades já foram calculadas pela detecção e pelo mapa de
desfecho; o custo por evento é **uma raiz quadrada e um produto vetorial**. O passe inteiro é
`O(n_events)`, não `O(n_events * N)`, e continua fora de `n_force`.

**O termo de terceiros omitido — medido, não estimado.** Estado real no pico de compressão,
`chi = 0.1`, `eps = 5.0e-2`, os `7` pares efetivamente em contato, `ΔU` exato pela soma `O(N)`
completa contra a forma fechada acima **[M]**:

| grandeza | valor |
|---|---|
| `E_grav` mútuo por fusão, `/ \|E_0\|` | `2.06e-4` |
| resíduo de terceiros por fusão, `/ \|E_0\|` | `-2.5e-6` |
| resíduo / termo mútuo | `1.2%`, de **sinal único** (negativo) |
| acúmulo sobre `~500` eventos, `/ \|E_0\|` | **`1.2e-3`, isto é `0.12%`** |

**Portanto `E_total = K + U + E_int` NÃO é exatamente conservada pelo mapa: ela deriva `~0.12%` de
`|E_0|` ao longo de uma execução de `3 t_ff`, por omissão declarada dos termos de terceiro corpo.**
Isso é o preço aceito, e é `~30x` menor que o degrau de um único evento.

**Rótulo obrigatório no HUD e em toda figura:**
`E_total = K + U + E_int (colisoes contabilizadas ao nivel do par; deriva residual ~0.1% |E_0|)`.
É **proibido** rotular esta curva como energia exatamente conservada.

**O termo mútuo, esse, NÃO pode sair.** Ele vale `2.06e-4 |E_0|` por fusão e tem sinal único;
omiti-lo (isto é, fazer `E_int += T_cm` apenas) fabricaria `~10%` de `|E_0|` de dissipação inexistente
ao longo de `500` eventos. **[M]** Isso é uma curva de energia visivelmente indo embora no HUD — um
defeito visual, não só de rigor. Ver Seção 4.14, item 7.

**Quantidades conservadas por construção:**

```
L_total = L_orb + L_spin         conservada exatamente pelo mapa de colisao        [T]
E_total = K + U + E_int          conservada a menos do residuo de terceiros, ~0.12% |E_0|/execucao [M]
```

**Aviso normativo sobre o que essas identidades testam.** `L_total` é, por construção, uma
**tautologia no evento**: não testa a física do desfecho, testa apenas que o acumulador foi somado.
O que testa a física são os invariantes analíticos **por desfecho** de `INV-20`, `INV-21`, `INV-22`
(`ΔK = -T_cm` na fusão, `ΔT_cm = 0` na fragmentação, `ΔK = ΔL = 0` na elástica). Os dois conjuntos
são complementares e **ambos** obrigatórios; nenhum substitui o outro.

`E_total`, após esta revisão, **deixou de ser tautologia**: o resíduo de terceiros é real e o teste
por evento (`INV-23(a)`) passa a medir se ele está dentro da banda prevista. Isso é uma melhora
acidental do poder discriminante — mas o teste que pega termo esquecido ou sinal trocado continua
sendo `INV-21`/`INV-22`, não `INV-23(a)`.

**`E_int` é um acumulador com sinal, e o sinal é um diagnóstico.**

- **~~Fusão sempre incrementa `E_int` por um valor não negativo.~~ RETRATADO em 2026-08-07 (b).**
  A justificativa era "a conservação de energia do par que cai **desde separação grande** garante
  `T_cm >= E_grav` no contato" — **a mesma premissa falsa da Seção 4.6.1**, aparecendo pela terceira
  vez no documento. No núcleo suavizado o par chega com apenas `~40%` dessa energia, logo
  `T_cm < E_grav` é possível e comum para pares desiguais. Calculado **[T]**:

  | par | `T_cm` | `E_grav` | `E_int +=` |
  |---|---|---|---|
  | `m_bar`–`m_bar` | `5.91e9 J` | `1.31e9 J` | `+0.0007 \|E_0\|` — dissipa |
  | `300 m_bar`–`m_bar` | `9.60e10 J` | `3.17e11 J` | **`-0.0344 \|E_0\|` — INJETA** |

  **A fusão pode injetar energia mecânica, e isso é uma propriedade do modelo, não um defeito de
  implementação.** Fisicamente: fundir dois corpos apaga o poço mútuo, e se o par não trouxe energia
  cinética suficiente para "pagar" esse poço, a diferença aparece como energia mecânica criada.
  ~~É por isso que `INV-23(c)` existe — e é por isso que ele não pode ser uma cota apertada.~~
  **Emendado em 2026-08-08 (d):** esta frase sobreviveu à retratação de mais abaixo, que retirou
  `INV-23(c)` da condição de cota por inteiro. A leitura correta é a oposta: a injeção legítima é
  uma das razões pelas quais **nenhuma** cota sobre `E_int` é um critério válido, apertada ou
  frouxa. O que resta de `INV-23(c)` é um número reportado.
- **Fragmentação tem sinal indefinido em geral**, com o sinal corrigido:
  `E_grav(m_a,m_b,R_a+R_b) - E_grav(m_i,m_j,d_ij)`. O primeiro termo cresce quando a partição de
  massa é equilibrada, o segundo quando `d_ij` é pequeno (contato profundo). Nem
  `m_a m_b <= m_i m_j` nem `R_a+R_b >= R_i+R_j` valem universalmente. **[T]**

  **Caso patológico, que é a segunda face do runaway.** Fragmentar um par **muito desigual** produz
  dois fragmentos **comparáveis**, e o produto `m_a m_b` explode. Para `300 m_bar + 1 m_bar` com
  `f = 0.5` **[T]**: `m_a m_b / (m_i m_j) = 75.5x`, e o incremento é `+3.17 |E_0|` **em um único
  evento**. Isso não é um defeito próprio: é o que acontece quando existe um corpo de `300 m_bar`.

  > **RETRATAÇÃO 2026-08-08 (d) — a frase que seguia aqui era sobrevivência da revisão (a).**
  > O texto dizia: *"Contido o runaway (Seção 4.6), o maior corpo fica em `~3 m_bar`, os pares
  > ficam quase iguais, e a patologia desaparece sozinha"*. **O runaway não foi contido e não é
  > contível neste modelo** (Seções 4.12, 4.13.4): a fragmentação é `2 -> 2`, conserva
  > `m_i + m_j`, não há sumidouro de massa e portanto não há teto. Medido `max_i m_i = 321.26
  > m_bar` **[M]**, em duas execuções independentes e com duas definições diferentes de `x`.
  >
  > **E há uma inversão a registrar, porque ela é o ponto.** O "caso patológico" descrito acima
  > não é uma consequência do runaway: **é o mecanismo que o produz.** A mesma aritmética que faz
  > `m_a m_b` explodir — fragmentar um par desigual devolve dois fragmentos comparáveis — é a que
  > **eleva** o máximo de massa sempre que o parceiro passa de `42.9%` do corpo grande em esperança
  > (Seção 4.13.4). O documento tinha o mecanismo escrito, em `E_int`, duas revisões antes de o
  > identificar em massa. **Lição transferível: quando um termo de energia explode com a
  > desigualdade de massa do par, a variável que explodiu antes foi a massa.**
  >
  > **O que NÃO muda:** este documento continua **não** acrescentando uma regra de razão de massa
  > à fragmentação — mas por outra razão, que não é mais "a patologia desaparece sozinha". A razão
  > vigente é a da Seção 4.13.5: conter exigiria fragmentação `2 -> muitos` ou supressão dependente
  > de massa, ambas **acrescentam** complexidade, e o comportamento foi **aceito como resultado**.
- `E_int(t)` pode legitimamente ficar negativa, e o sinal **deixou de ser um critério**.

**`INV-23(c)`, 2026-08-08: REPORTADO, NÃO BLOQUEANTE. Medido `10.83 |E_0|`.**

> **Terceira e última formulação, e a mudança é de tipo, não de valor.** As três versões anteriores
> (`>= -1e-3`, `>= -1e-2`, `|E_int| <= 1`) foram todas **cotas**, e as três reprovaram. Elevar a cota
> uma quarta vez seria exatamente o ajuste post-hoc que este documento proíbe.
>
> O diagnóstico correto é que **`|E_int| >> |E_0|` não é um defeito: é a assinatura quantitativa do
> runaway de fusão** (Seção 4.13.4), que o projeto agora **aceita como resultado** (Seção 4.13.5).
> Um número que mede fielmente um fenômeno aceito não pode ser critério de reprovação desse
> fenômeno. `E_int(t)/|E_0|` passa a ser **reportado com sinal**, exibido no HUD e no relatório como
> curva, e **nenhum teste falha por causa do seu valor**.
>
> **O que o rótulo tem de dizer, obrigatoriamente:** quando `|E_int| > |E_0|`, o livro de colisões
> movimentou mais energia do que existe ligando o aglomerado, e a curva de `E_total` **deixa de ser
> um diagnóstico do integrador** — passa a ser dominada pelo modelo de colisão. Afirmar qualquer
> coisa sobre qualidade de integração a partir dela, nesse regime, é proibido.

**Cota antiga, preservada para leitura da história:** `max_t |E_int(t)| <= 1.0 |E_0|`.

> **[HISTÓRICO — revisão (b), 2026-08-07. NÃO É NORMATIVO.]** O bloco abaixo justifica a segunda
> das três formulações reprovadas de `INV-23(c)` e está preservado porque explica por que a cota de
> **sinal** caiu. Ele foi escrito antes da retratação acima e usa o presente do indicativo ("é esse
> o critério"); leia-o como registro. **O critério vigente está na caixa `INV-23(c), 2026-08-08`
> acima: não há cota.** Marcador acrescentado em 2026-08-08 (d), porque sem ele o bloco lia como
> norma em vigor.

> **A cota deixou de ser sobre o SINAL e passou a ser sobre a MAGNITUDE.** As duas versões anteriores
> (`>= -1e-3`, depois `>= -1e-2`) guardavam o sinal, sob a premissa de que `E_int < 0` denunciaria
> injeção espúria. A retratação acima mostra que **a fusão injeta legitimamente** neste modelo, e que
> o incremento de um único evento (`0.034 |E_0|`) já excede qualquer cota de sinal por ordens de
> grandeza. **Uma cota de sinal reprovaria uma implementação correta.**
>
> O que o critério precisa pegar de fato é *"as colisões passaram a ser a física dominante"*, e a
> forma honesta disso é comparar o livro de colisões com a energia de ligação do sistema:
> `|E_int| > |E_0|` significa que o modelo de colisão movimentou mais energia do que existe ligando
> o aglomerado. É esse o critério, e é ele que o `-109.8` medido violou por duas ordens.

`E_int(t)/|E_0|` **deixou de ser guarda em 2026-08-08 e é hoje apenas** um **resultado físico por
direito próprio**: é o orçamento de dissipação do colapso colisional, e deve ir para o relatório
como curva, com sinal e com `t_runaway` marcado. *(A redação anterior — "além de guarda" — era
sobrevivência da revisão (b) e está emendada em 2026-08-08 (d): não há mais o "além".)*

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

**Fusão/fragmentação.** `E_mec = K + U` exibe uma **escada**: cada evento é um degrau finito.

> **EMENDA 2026-08-08 (d) — a magnitude do degrau estava desatualizada.** O texto dizia "de até
> `~1e-3 |E_0|`", número da revisão (a), quando ainda se supunha um teto de massa em `~3 m_bar`.
> Com o runaway aceito, a mesma seção 4.10 registra degraus de `-0.0344 |E_0|` (fusão `300:1`) e
> `+3.17 |E_0|` (fragmentação `300:1`) **[T]**, três e seis ordens acima. **A escala do degrau não
> é uma constante do modelo: ela escala com `m_i m_j` do par**, e por isso cresce com o corpo
> dominante. Enunciado correto: o degrau é limitado por `E_grav(m_i, m_j, d_ij) + T_cm`, que não
> tem cota superior neste modelo porque a massa não tem.

Não há banda limitada, não há oscilação em torno de valor fixo, e a razão
"pico/final ≈ 60" que distingue simpléticos de não simpléticos em `INV-4` **deixa de existir como
teste**. Qualquer critério de `INV-4` aplicado a `E_mec` numa execução com fusão falha contra uma
implementação correta.

**Diagnóstico substituto — quatro peças, todas obrigatórias:**

- **(D1) `|Delta E_total / E_total(0)|` com `E_total = K + U + E_int`.** É o sucessor direto da
  banda antiga. Os critérios **qualitativos** de `INV-4` (`velocity_verlet` não monótono, final
  muito menor que o pico; `euler` monótono crescente; `rk4` derivando negativo) transferem-se para
  `E_total`. Os valores **[M]** de `integradores.md` **não** se transferem: a trajetória é outra.

  > **DUAS CORREÇÕES 2026-08-08 (d). O texto anterior dizia "conservado exatamente pelos eventos,
  > portanto sua deriva mede só o integrador entre eventos", e as duas metades da frase estão
  > erradas — a primeira desde a própria revisão (a), na seção seguinte à que a escreveu.**
  >
  > 1. **`E_total` NÃO é conservada exatamente pelos eventos.** A Seção 4.10 declara que os termos
  >    de terceiro corpo são deliberadamente omitidos e que o resíduo é de **sinal único**:
  >    `2.5e-6 |E_0|` por evento, `~1.2e-3 |E_0|` acumulados em `3 t_ff` **[M]**, no regime de
  >    massas comparáveis. `(D1)` mede o integrador **mais** esse resíduo, e o resíduo é
  >    sistemático, não ruído.
  > 2. **No regime pós-runaway `(D1)` não mede o integrador de forma alguma.** Medido
  >    `|ΔE_total/E_0| = 8.59` contra `max_t |E_int|/|E_0| = 10.83` **[M]**: a "deriva" é
  >    `79%` da magnitude do próprio livro de colisões. Não há como atribuir isso ao truncamento
  >    de Verlet, cujo pico sem colisões é `2.288e-4` **[M]**, quatro ordens abaixo.
  >
  > **Regime de validade de `(D1)`, normativo:** enquanto `max_i m_i / M_real < 0.10`, e ainda
  > assim com o piso sistemático de `~1.2e-3 |E_0|` declarado. Além de `t_runaway`, `(D1)` é
  > **reportado** e é proibido dele extrair afirmação sobre o integrador (Seção 4.10, rótulo
  > obrigatório; Seção 4.13.5, obrigação de marcar `t_runaway`). A comparação entre integradores é
  > feita **sem colisões**, como as proibições explícitas ao fim desta seção já exigiam.
- **(D2) `E_int(t)/|E_0|`.** O orçamento de dissipação. Resultado físico; **não é guarda** — a
  formulação "não só guarda" era da revisão (b) e está emendada em 2026-08-08 (d), ver 4.10.
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

> # ERRO DE ANÁLISE, IDENTIFICADO EM 2026-08-07 (b)
>
> **Esta seção concluiu "não há crescimento descontrolado" e a execução do estágio 3 encontrou
> `max m_i = 321 m_bar`, `90x` a previsão.** O erro está localizado, e em uma linha é este:
>
> > **A Seção 4.12 resolveu o ponto fixo de `dm/dt` mantendo `p_fus` e `p_frag` congelados num único
> > `x` "típico do núcleo", quando `x` é ele próprio uma função decrescente da massa do corpo em
> > crescimento, através de `v_esc_eff² ≈ 2GM/eps`.**
>
> A fórmula do teto, `m*/m_bar = (p_fus + k p_frag)/((1-k) p_frag)`, é **correta como ponto fixo a
> `p` constante**. O que não foi feito foi verificar que o mapa `m -> m*(m)` tem um **cruzamento
> estável**. Não tem: `m*(m)` cresce mais depressa que `m`, e o teto **foge à frente da massa** em
> vez de contê-la. Iterando `m <- m*(m)` a partir de `m = 3` **[T]**:
>
> | `m` | `3.00` | `5.55` | `7.82` | `10.3` | `13.6` | `18.7` | `28.4` | `52.6` | `144` | `826` | `1.7e4` |
> |---|---|---|---|---|---|---|---|---|---|---|---|
> | `m*(m)` | `5.55` | `7.82` | `10.3` | `13.6` | `18.7` | `28.4` | `52.6` | `144` | `826` | `1.7e4` | `1.8e6` |
>
> **Não há ponto fixo. O "teto fechado" que esta seção anunciou não existe** sob a definição de `x`
> que vigorava. A causa está identificada na Seção 4.6.1 e corrigida na Seção 4.6: retirado
> `v_esc_eff`, `x` deixa de depender da massa, **e só então a derivação abaixo passa a ser válida**.
>
> **Registrar isto é o ponto.** Errar uma previsão é barato; não registrar por que se errou é caro. A
> lição transferível: um ponto fixo calculado com os coeficientes congelados só é um teto se os
> coeficientes não dependerem da variável de estado. Verificar essa dependência é obrigatório, e não
> foi feito.

A preocupação é legítima e o cálculo a responde — **desde que `x` não dependa da massa**, o que a
Seção 4.6 revisada agora garante. Há dois laços de realimentação em sentidos opostos.

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

A variação esperada de massa por evento, **para um parceiro de massa `m'`**, é

```
E[Delta m] = p_fus * m'  +  p_frag * [ k (m + m') - m ]
           = p_fus * m'  +  p_frag * [ k m' - (1-k) m ]
```

que se anula em

```
m* / m'  =  ( p_fus + k p_frag ) / ( (1-k) p_frag )                                [T]
```

> # SEGUNDO ERRO DESTA SEÇÃO, IDENTIFICADO EM 2026-08-08
>
> **O denominador da fórmula é `m'`, a massa do PARCEIRO — não `m_bar`.** Esta seção substituiu
> `m' = m_bar` e leu o resultado como um teto absoluto em unidades de `m_bar`. Ele nunca foi isso:
> é um teto **relativo ao parceiro típico**. Assim que a fragmentação cria uma segunda população
> massuda, `m'` cresce, e **o teto cresce proporcionalmente com ela**.
>
> Com as frações de canal **medidas** (`p_fus = 0.069`, `p_frag = 0.650`): `m* = 2.69 m'` **[M]**.
> Um teto que vale `2.69x` o parceiro não contém nada quando o parceiro também cresce.
>
> **É o mesmo erro da revisão (b), pela terceira vez:** um ponto fixo calculado com um coeficiente
> co-evolutivo mantido fixo. Na revisão (b) o coeficiente congelado era `x`; aqui é `m'`.
>
> **E há uma razão estrutural por baixo, que nenhuma correção de fórmula resolve.** A fragmentação é
> `2 -> 2` e conserva `m_i + m_j` **exatamente**: ela **não retira massa do conjunto**, só
> redistribui. O único canal que muda a massa detida pela população grande é a fusão, que só
> concentra. **Este modelo não tem sumidouro de massa, portanto não tem teto de massa — para nenhum
> valor de nenhum parâmetro.** **[T]**
>
> Além disso, a fragmentação **eleva** o máximo sempre que o parceiro passa de `(1-k)/k = 42.9%` do
> corpo grande em esperança (e de apenas `11.1%` quando `f` sai em `0.9`). **[T]** Ela é um canal de
> concentração, não de contenção.
>
> **Consequência normativa: o "teto fechado" abaixo está RETRATADO.** Os números que seguem
> permanecem apenas como o cálculo de campo médio que foi feito e falhou, e como registro do erro.
> **Não são predição.** A predição vigente é a da Seção 4.13.5: **não há teto**, e o valor final é
> fixado pela massa disponível na vizinhança do corpo, não pelo mapa.

**~~Este é um teto fechado, e é um ponto fixo atrator.~~ RETRATADO** — ver a caixa acima. Valores
do cálculo de campo médio, preservados como registro **[T]**:

| caso | `p_fus` | `p_frag` | `m*/m_bar` | `m*/M_tot` |
|---|---|---|---|---|
| controle uniforme `1/3` | `0.3333` | `0.3333` | `5.667` | `0.57%` |
| **mapa `(1/x,3,x)/Z`, `x = \|u\|²/v_coh² = 2.00` no núcleo** | `0.0908` | `0.3638` | **`3.17`** | **`0.32%`** |
| (histórico) softmax `w=3`, `x = 1.450` | `0.2555` | `0.3274` | `4.935` | `0.49%` |

> **[HISTÓRICO. NÃO É NORMATIVO.]** Os três parágrafos que seguem — o "teto baixa de `4.94` para
> `3.53`", a integração da EDO de campo médio e a sensibilidade em `v_coh` — falam de um **teto que
> não existe** e estavam redigidos no presente do indicativo. Marcador acrescentado em 2026-08-08
> (d). Registre-se ainda uma **incoerência interna do próprio cálculo histórico**: o texto diz
> `3.53 m_bar` enquanto a tabela imediatamente acima dá `3.17` para o mesmo caso. Nenhum dos dois
> valores é predição; **não foram reconciliados de propósito**, porque reconciliar números de um
> cálculo retratado é dar-lhes um estatuto que eles não têm.

**~~O mapa simplificado BAIXA o teto de massa~~**, de `4.94` para `3.53 m_bar`. A conclusão
qualitativa não muda e fica mais folgada: com `p_fus` menor, o laço positivo é mais fraco e o teto
desce. O controle uniforme (`5.667`) é independente do mapa e permanece como está.

Integração da EDO de campo médio, com a taxa da Seção 4.1 e o mapa da Seção 4.7 **[T]**:

| `chi` | `m/m_bar` em `0.21 s` | em `0.63 s` | `x` final | `p_fus` | `p_frag` |
|---|---|---|---|---|---|
| `1` | `4.70` | `4.70` | `1.669` | `0.243` | `0.342` |
| `0.5` | `4.82` | `4.84` | `1.537` | `0.250` | `0.333` |
| `0.25` | `3.67` | `4.90` | `1.452` | `0.255` | `0.327` |
| `0.1` | `1.48` | `2.47` | `1.544` | `0.250` | `0.334` |

A tabela acima foi computada com o softmax `w = 3` e permanece como referência histórica; com o mapa
`(1/x, 3, x)/Z` os tetos descem `~30%`, para `~3.5 m_bar`.

Sensibilidade: `v_coh = 2 V_CHAR` dá `m* = 6.78 m_bar`; `3 V_CHAR` dá `8.88`. **[T]** As entradas de
sensibilidade em `w` foram **retiradas**: `w` não existe mais. A única alavanca de forma que restou é
`v_coh` (Seção 4.6), e é ela que domina a sensibilidade do teto.

**~~Veredito: não há crescimento descontrolado, e o maior corpo satura em `~3.2 m_bar`.~~
RETRATADO EM 2026-08-08 — MEDIDO `321.26 m_bar`, `100x` acima.** **[M]**

A revisão (b) tornou `p_fus` e `p_frag` independentes de `m`, o que corrigiu **um** dos dois defeitos
desta seção. O que restava, e continua valendo, é que **`m*` está em unidades do parceiro `m'`, não
de `m_bar`**, e que a fragmentação não é um sumidouro de massa. **Não há teto.** O parâmetro
`a = FRAG_F_MIN` modula `k = 3/4 - a/2` e portanto a *velocidade* do crescimento, mas não cria um
teto onde a estrutura do modelo não permite um. Ver Seção 4.13.4, e a decisão de aceitar em 4.13.5.

Taxa de despovoamento, com `chi = 0.1` e o mapa novo (`p_fus = 0.114` no núcleo): `~25` fusões
durante o primeiro rebote, contra `500` corpos no núcleo — **`5%` do núcleo, `2.5%` da população**,
por rebote. (Com o softmax antigo eram `~56`.) **[T]** Este cálculo, e não uma preferência estética,
é o que sustenta `chi = 0.1` da Seção 4.1 — e a medição de 4.1.1 o confirmou.

### 4.13 O invariante de ensemble

O critério de aceitação **não é** "não pode haver runaway" — runaway pode ser a física correta. O
critério é sobre a **degenerescência**: rejeita-se o modelo se toda semente terminar em um corpo, e
depressa.

**Protocolo (normativo, revisado).** **`K_SEEDS = 4`** execuções do estágio 3, `t_end = 3 t_ff`,
variando apenas `mass_seed` e a semente de colisão; posições e `Q` fixos.

> **Reduzido de `32` para `4` nesta revisão.** A justificativa antiga era estatística: com `0/32`
> sementes degeneradas, a regra de três limita a taxa verdadeira a `3/32 = 9.4%` a `95%` de
> confiança, casando com o limiar de `10%` de (C1). Com `4` sementes essa resolução cai para
> `3/4 = 75%`, e **(C1) deixa de ser um teste estatístico útil** — passa a detectar apenas a
> degenerescência grosseira ("toda semente termina em um corpo"), que é o que o critério de fato
> existe para pegar. Sob o critério deste projeto, essa é a troca certa: `32` sementes custariam
> `~20 min` de campanha para resolver uma taxa de degenerescência que, se for não nula, será
> visível em `4`. **O que se perde está dito com todas as letras: (C1) e (C4) deixam de ter poder
> estatístico e viram inspeções.** (C4), o índice de dispersão, é **retirado da lista de critérios**
> (ver abaixo) porque uma variância sobre `4` amostras não é uma medição.

Grandezas registradas por semente: `N_final` (corpos vivos em `3 t_ff`), `t_50` (primeiro instante
com `N_live <= N/2`, ou `> 3 t_ff`), `n_merge`, `n_frag`, `n_elastic`, `max_i m_i / M_real`,
`min_i m_i / m_bar`, `min_t E_int/|E_0|`, `f_reject_max`, `f_reject_total`, `c_coll_max`.

#### 4.13.1 Predição falsificável, registrada ANTES da implementação

Esta é a predição de conteúdo do estágio 3, escrita antes de existir uma linha de `resolve()`,
precisamente para que ela possa falhar.

**Derivação.** A linha de base do estágio 2 mediu `1902` encontros ao longo de `3 t_ff` em
`chi = 0.1`, massas iguais, `Q = 0` **[M]**. Cada fusão retira **um** corpo vivo. Os encontros do
núcleo são dominados por pares `m_bar–m_bar` a `|u| ~ 4.6 m/s`, isto é `x ≈ 1.67`, onde o mapa da
Seção 4.7 dá `p_fus = 0.114`. Tomando `<p_fus> ≈ 0.12` sobre a população visitada:

```
n_merge  ~  1902 * 0.12  ~  230           =>   N_final ~ 1000 - 230 = 770
```

Duas correções de segunda ordem em sentidos opostos, ambas de ordem `10%` e que não se resolvem sem
medir: o despovoamento **reduz** a taxa ao longo da execução; corpos fundidos têm `R` maior por
`2^(1/3) = 1.26`, o que **aumenta** a secção de choque. **[A]**

**PREDIÇÃO NORMATIVA (`chi = 0.1`, massas iguais, `Q = 0`, `dt = 5.0e-4`, `3 t_ff`):**

```
N_final  ∈  [700, 800]                de 1000            [A]
t_50     >  3 t_ff   (isto e: N_live NUNCA cai a 500)    [A]
max_i m_i / m_bar   ~  3.5,  e  < 10                     [A], da Secao 4.12
sem runaway: max_i m_i / M_real  <  0.01                 [A]
```

**Como ler uma falha — isto é o que dá valor à predição:**

| observado | leitura |
|---|---|
| `N_final ∈ [700, 800]` | modelo se comporta como especificado |
| `N_final > 950` | os eventos não estão disparando: conferir detecção, `chi`, ou o desvio de canal |
| `N_final < 500` | `<p_fus>` está sendo entregue muito acima de `0.12`: conferir o cálculo de `x` (suspeito principal: `E_bind` sem o termo gravitacional, ou `v_coh` errado) |
| `N_final = 1` ou próximo | degenerescência: (C1) reprova, reduzir `chi` |

A predição é para **massas iguais**. Com o espectro de massas ligado, os corpos leves têm `x` mais
baixo (Seção 4.8) e portanto mais fusão: espera-se `N_final` **menor**, e a banda `[700, 800]` **não
se aplica** a essa população. **[A]**

#### 4.13.2 Resultado do estágio 3 — a predição FALHOU, de forma diagnóstica

Executado 2026-08-07, `N = 1000`, `chi = 0.1`, `dt = 5e-4`, `Q = 0`, `3 t_ff`, fp64. **[M]**

| grandeza | previsto | medido | veredito |
|---|---|---|---|
| `N_final` | `[700, 800]` | `744` | **bate** |
| `t_50` | `> 3 t_ff` | nunca | **bate** |
| canais (el/fus/frag) | cada `>= 5%` | `34.5% / 12.9% / 52.7%` | **(C5) passa** |
| massa | exata | desvio `2.4e-16` em `12601` passos | **exata** |
| momento | exato | exato | **exato** |
| `max m_i / m_bar` | `~3.5`, `< 10` | **`321.4`** | **FALHA, `90x`** |
| `max m_i / M_real` | `< 0.01` | **`0.3214`** | **FALHA — é runaway** |
| `min_t E_int/\|E_0\|` | `>= -1e-2` | **`-109.8`** | **FALHA, `4` ordens** |
| `\|ΔE_total/E_0\|` | `<= 1e-2` | **`107.5`** | **FALHA, `4` ordens** |

Sem `NaN`, sem `Inf`, estado finito o tempo todo. **Não é corrupção numérica.** Crescimento suave e
comportado até `t ≈ 1.55 t_ff` com `max m ≈ 7 m_bar` — exatamente o regime que a Seção 4.12 previu —
e então explosão em `~150` passos, de `7` para `128 m_bar`, seguindo até `321`.

**Três vereditos distintos, que não devem ser confundidos:**

1. **A IMPLEMENTAÇÃO está vindicada.** Massa e momento conservam-se exatamente, não há `NaN`, e a
   trajetória segue a predição do documento até o ponto em que a predição deixa de valer. Os itens do
   PISO que protegem massa e momento funcionaram. **Nada aqui aponta para defeito de código.**
2. **A PREDIÇÃO estava errada**, e o erro de análise está localizado e registrado na Seção 4.12.
3. **O MODELO estava defeituoso**, e isso é o principal: a dependência de `x` com a massa via
   `v_esc_eff` era um erro de modelagem, não um erro de previsão. Corrigido na Seção 4.6.

**`E_int = -109.8` tem DUAS causas, e ambas precisavam aparecer:**

- **Um erro de sinal deste documento** na expressão de fragmentação da Seção 4.10, fielmente
  implementado. Corrigido lá. Sozinho, ele inverte o sinal do termo dominante.
- **A magnitude é consequência do runaway.** Fragmentar `300 m_bar + 1 m_bar` multiplica `m_a m_b`
  por `75.5x` e move `3.17 |E_0|` **num único evento** (Seção 4.10). Sem corpos massudos, esse termo
  é pequeno. **Corrigido o runaway, a magnitude colapsa junto.** Não há um terceiro problema
  escondido: os dois canais de falha compartilham a mesma raiz — a premissa de queda desde o
  infinito, que a Seção 4.6.1 refuta.

**`321 m_bar` é "rápido demais"? Sim — e a pergunta é sobre a tela, não sobre rigor.** Três fatos
visuais e estruturais, nenhum deles uma violação de conservação:

1. **`33%` da massa num corpo** faz a cena virar uma esfera grande e `743` pontinhos. Uma população
   graduada de massas é mais interessante de ver que um dominante.
2. **A transição toma `~1%` do tempo de execução** (`150` de `12601` passos). Na tela isso lê como
   *glitch*, não como física. Um runaway que fosse o resultado seria gradual.
3. **O modelo perde a sua própria autoconsistência.** `R = 0.034 m` para o corpo grande aproxima-se
   de `eps = 0.05 m` e é `10%` de `r_half,min = 0.35 m`: a "partícula" tem o tamanho de um décimo do
   núcleo, e a descrição "massa pontual + softening" deixa de valer para ela. `V_CHAR`, `L_SCALE` e a
   própria escolha de `eps` supõem que nenhum corpo domina — é exatamente o que `INV-31(C7)` dizia.

**E o argumento decisivo sob o critério deste projeto: a correção REDUZ complexidade.** Retirar
`v_esc_eff` apaga um termo, uma raiz e uma soma por evento. Não há trade-off a fazer aqui: o modelo
mais simples é também o que não tem o defeito.

#### 4.13.3 Nova predição falsificável, registrada ANTES da próxima execução

Com `x = |u|²/v_coh²` (Seção 4.6) e o sinal de fragmentação corrigido (Seção 4.10), mesmas condições:

```
max_i m_i / m_bar    ∈  [2, 8]        centrado em 3.17, da Sec. 4.12 agora valida   [A]
max_i m_i / M_real   <  0.01                                                        [A]
N_final              ∈  [750, 900]    p_fus cai de 0.129 para ~0.09, menos fusoes   [A]
t_50                 >  3 t_ff                                                      [A]
max_t |E_int|/|E_0|  <  1.0           INV-23(c) reformulado                         [A]
canais               cada um >= 5%    RISCO: a fusao e' o canal em risco, Sec. 4.8   [A]
```

**Leitura das falhas:**

| observado | leitura |
|---|---|
| `max m_i` ainda `>> 10 m_bar` | sobrou dependência de `x` com a massa em algum ramo — conferir se `v_esc_eff` foi de fato retirado |
| canal de fusão `< 5%` | a distribuição de `\|u\|` no núcleo é mais dura que o previsto; alavanca declarada: **reduzir** `v_coh` (Seção 4.6.2) |
| `E_int` grande e **negativo** | o sinal de fragmentação não foi corrigido |
| `N_final > 950` | eventos não disparando; conferir detecção |

#### 4.13.4 Resultado da revisão (b) — a predição falhou de novo, e aqui o modelo PARA de ser mexido

Executado 2026-08-08, mesmos parâmetros. **[M]**

| grandeza | previsto (4.13.3) | medido | veredito |
|---|---|---|---|
| `N_final` | `[750, 900]` | `774` | **bate** |
| `t_50` | `> 3 t_ff` | nunca | **bate** |
| `max m_i / m_bar` | `[2, 8]` | **`321.26`** | **falha** |
| `max m_i / M_real` | `< 0.01` | **`0.3213`** | **falha** |
| `max \|E_int\|/\|E_0\|` | `< 1` | **`10.83`** | **falha** (era `109.8`) |
| `\|ΔE_total/E_0\|` | — | `8.59` | era `107.5` |
| canais | cada `>= 5%` | `28.1 / 6.9 / 65.0 %`, `3280` eventos | **(C5) passa** |
| massa, momento | exatos | `2.4e-16` em `12601` passos | **exatos** |

**As correções da revisão (b) funcionaram**: `E_int` melhorou `10x`, a energia total `12x`. E, o que
importa mais, **o `x` algébrico não depende mais de massa nenhuma** — e o runaway ocorreu assim
mesmo, com valor final praticamente idêntico (`321.26` contra `321.4`). Existe um **segundo
mecanismo**, e ele não está na fórmula do mapa.

**O argumento que decide, e não precisa de execução nova — é contagem.**

```
fusoes na execucao inteira            = 6.9% de 3280 = 226
queda de N: 1000 -> 774               = 226           (confere: cada fusao mata um slot)
arvore binaria de fusoes necessaria
para montar 321 m_bar a partir de
corpos de 1 m_bar                     = 320 fusoes

320 > 226   ->   IMPOSSIVEL construir o corpo por fusao.                           [T]
```

**Portanto a fragmentação AUMENTOU a massa de algum corpo. Isso não é hipótese: é aritmética sobre
os números já medidos.**

**Veredito sobre as duas hipóteses:**

- **(b), do coordenador — NECESSÁRIA, e mais forte do que foi enunciada.** Foi enunciada como
  "a fragmentação não desfaz o crescimento quando o par é muito desigual". Medido, o enunciado
  literal está **errado**: fragmentar `(300, 1)` dá produto máximo esperado `0.70 x 301 = 211`, que é
  uma **redução** de `30%`. O correto é mais grave: **a fragmentação AUMENTA o máximo sempre que o
  parceiro passa de `(1-f)/f` do corpo grande** — para `f = 0.9`, apenas `11.1%`; em esperança
  (`k = 0.70`), **`42.9%`**. **[T]** Ela não é um canal de contenção que falha; é um canal de
  **concentração de massa** que funciona.
- **(a), do implementador — não é necessária, e não pode ser dominante.** Ela elevaria `p_fus` nos
  encontros do corpo massudo. Mas **mesmo com `p_fus = 1` nas 226 fusões existentes**, `226 < 320`:
  a fusão não dá conta. (a) pode modular a taxa; não explica o resultado. Por Occam, fica registrada
  como **não verificada e não necessária**.
- **A causa estrutural, que nenhuma das duas nomeia:** a fragmentação é `2 -> 2` e conserva
  `m_i + m_j` **exatamente**. Logo **ela não pode retirar massa do conjunto de corpos envolvidos —
  só redistribui.** O único canal que altera a massa detida pela população grande é a fusão, e a
  fusão só concentra. **ESTE MODELO NÃO TEM SUMIDOURO DE MASSA.** Com qualquer `p_fus > 0`, o máximo
  cresce até esgotar a vizinhança. Nenhum valor de nenhum parâmetro muda isso.

#### 4.13.5 DECISÃO: parar de mexer no modelo e corrigir a PREDIÇÃO

**Recomendação: não há terceira rodada de modelagem. O comportamento é ACEITO e passa a ser relatado
como resultado.** Cinco razões, em ordem de peso:

1. **Não existe correção simples.** A causa é estrutural (ausência de sumidouro de massa), não
   paramétrica. Contê-la exigiria mudar a fragmentação de `2 -> 2` para `2 -> muitos`, ou introduzir
   supressão dependente de massa — **ambas acrescentam complexidade**, e a regra deste projeto é
   preferir remover a acrescentar. Não há o que remover aqui.
2. **O critério de aceitação declarado pelo usuário está satisfeito.** "Terminar em um corpo só não é
   falha, desde que não aconteça sempre nem muito rápido." Não termina em um corpo: **`774` corpos
   sobrevivem**. E não é rápido: a transição toma **`1.43 t_ff`, isto é `48%` da execução**, contra
   `1.2%` na rodada 1. **[M]**
3. **O que se vê na tela melhorou muito, e é o melhor produto disponível.** Comparação direta:
   - **rodada 1:** transição em `1.2%` do tempo — na tela lê como **glitch**;
   - **rodada 3 (atual):** colapso frio, rebote em `1.04 t_ff`, e a partir de `~1.5 t_ff` **um objeto
     cresce progressivamente ao longo de metade da execução**, comendo o núcleo, terminando dominante
     entre `773` corpos pequenos. Isso lê como **processo**, tem começo, meio e fim, e é
     reconhecivelmente um **runaway merger** — fenômeno real em aglomerados densos;
   - **alternativa "contida"** (se uma terceira rodada funcionasse): uma população de corpos entre `1`
     e `~5 m_bar`, visualmente quase uniforme, com lampejos ocasionais de fusão e quebra. **Menos
     interessante de ver.**
4. **Duas rodadas, dois mecanismos, cada um real e cada um descoberto só depois de eliminar o
   anterior.** Isso é a assinatura de um alvo (`max m ~ 3 m_bar`) que **nunca foi propriedade deste
   modelo** — era propriedade de uma aproximação de campo médio que falhou três vezes seguidas
   (Seção 4.12). Uma terceira rodada muito provavelmente encontraria um terceiro mecanismo.
5. **O documento já tinha escrito a regra certa, duas revisões atrás**, em `INV-31(C7)`: *"Se o
   runaway ocorrer, ele é o resultado e deve ser relatado como tal; o que não é permitido é que
   ocorra sem ser detectado."* Ele ocorreu, foi detectado, e está medido. **Honrar essa regra agora é
   mais consistente do que continuar lutando contra ela.**

**O que NÃO se aceita junto, e continua obrigatório declarar.** O runaway **invalida as escalas do
projeto** para `t > 1.5 t_ff`: `V_CHAR`, `L_SCALE` e a própria escolha de `eps` supõem que nenhum
corpo domina, e `R = 0.034 m` do corpo grande é `10%` de `r_half,min`. Toda figura e todo texto sobre
a fase colisional **têm de marcar `t_runaway`** e declarar que além dele o sistema não é mais o
colapso frio de N corpos parametrizado por essas escalas. Isso é `INV-31(C7)`, e ele é **reportagem
obrigatória**, não critério de reprovação.

#### 4.13.6 Predição vigente — do modelo COMO ELE É, não como se desejava que fosse

As três predições anteriores tentaram prever um **teto de massa**. Não existe teto (Seção 4.13.4), e
insistir seria prever contra a estrutura do modelo. A predição vigente descreve o runaway em vez de
negá-lo, e continua sendo falsificável.

**Predição (`chi = 0.1`, massas iguais, `Q = 0`, `dt = 5.0e-4`, `3 t_ff`, semente de posições fixa,
variando apenas a semente de colisão):**

```
N_final              ∈  [700, 900]                                                  [M]->[A]
t_50                 >  3 t_ff        (N_live nunca cai a 500)                       [A]
t_runaway            ∈  [1.2, 2.0] t_ff   (primeiro t com max m_i/M_real >= 0.10)     [A]
max_i m_i / M_real   ∈  [0.15, 0.60] em t = 3 t_ff                                   [A]
duracao da transicao >  0.5 t_ff      (isto e: >= 17% da execucao, NAO um degrau)     [A]
max_t |E_int|/|E_0|  ∈  [3, 40]                                                      [A]
canais               cada um >= 5%                                                   [M] 28.1/6.9/65.0
massa, momento       exatos a n_events * eps_prec                                    [T]
```

**As três que carregam conteúdo, e como falsificá-las:**

| se ocorrer | leitura |
|---|---|
| `t_runaway < 1.2 t_ff` ou transição `< 0.5 t_ff` | voltou a ser um degrau; na tela lê como glitch e o modelo precisa ser mexido |
| `max m_i / M_real > 0.6` em várias sementes | tende à degenerescência: `INV-31(C1)` volta a ser o critério, e a alavanca é reduzir `chi` |
| `max m_i / M_real < 0.05` | o runaway **não** é robusto entre sementes, e a Seção 4.13.5 decidiu com uma amostra de uma; reabrir |
| massa ou momento inexatos | defeito de implementação — nada nesta seção o autoriza |

**A predição mais importante é a de robustez entre sementes.** Toda a Seção 4.13.5 decide sobre
**uma** execução. `INV-31` com `K_SEEDS = 4` é exatamente o que a testa: se as `4` sementes
concordarem que há runaway lento e não degenerado, a decisão de aceitar está apoiada; se metade não
produzir runaway, o comportamento é bimodal e a decisão precisa ser revista. **Essa medição foi
feita — ver a Seção 4.13.7.**

**Critérios de aceitação (`INV-31`):**

- **(C1) Não degenerado no ponto final.** **Nenhuma** das `4` sementes termina com `N_final = 1`.
  Com `K = 4` isto não é um teste estatístico e não pretende ser: é uma inspeção de degenerescência
  grosseira.
- **(C2) Não degenerado do outro lado.** mediana de `n_merge + n_frag + n_elastic >= 50`. Zero
  eventos é tão degenerado quanto todos.
- **(C3) Não rápido demais.** `t_50 > 1.0 t_ff` em **todas** as `4` sementes. Se a população cai
  pela metade antes de o colapso completar, o modelo de colisão substituiu a física que estava sendo
  demonstrada em vez de decorá-la. Pela predição de 4.13.1, espera-se `t_50 > 3 t_ff`, de modo que
  este critério deve passar com folga larga; passá-lo raspando já é sinal de que `<p_fus>` está alto
  demais.
- **~~(C4) Índice de dispersão.~~ RETIRADO nesta revisão.** Com `K = 4`, `Var(n_merge)` sobre quatro
  amostras não é uma medição de dispersão, e um critério construído sobre ela reprovaria ou aprovaria
  ao acaso. O sintoma que (C4) existia para pegar — bimodalidade, "algumas sementes disparam, outras
  não" — passa a ser detectado por inspeção direta de `n_merge` das `4` sementes, reportado como
  lista e não como estatística. **Isto é uma perda real de poder de detecção, e está declarada.**
- **(C5) Mapa exercitado.** Cada canal recebe `>= 5%` dos eventos, agregado sobre o ensemble.

Todos os limiares acima são **[A]** e são critérios de **projeto**, não medições. Se o estágio 3
falhar (C1) ou (C3), a correção é reduzir `chi`; se falhar (C2), aumentar `chi`; **se falhar (C5)
pelo lado da fusão, a única alavanca autorizada é elevar `v_coh` acima de `V_CHAR` (Seção 4.6)** — o
mapa não tem mais parâmetros de forma para mexer. Nenhum deles pode ser afrouxado depois de ver o
resultado — ver a regra do cabeçalho de `tests/tolerances.py`.

#### 4.13.7 Resultado do ensemble `K_SEEDS = 4` — a predição vigente BATE

Executado 2026-08-08. **[M]** Protocolo exatamente o da Seção 4.13.6: `N = 1000`, `chi = 0.1`,
`dt = 5.0e-4`, `Q = 0`, `3 t_ff`, fp64, massas iguais, **semente de posições FIXA em `config.SEED`**,
variando **apenas** a semente de colisão, `COLLISION_SEED + k` para `k ∈ {0,1,2,3}`.

| semente | `N_final` | `max m / M_real` | `t_runaway / t_ff` | duração `/ t_ff` | `\|E_int\|/\|E_0\|` |
|---|---|---|---|---|---|
| `20190225` | `774` | `0.3213` | `1.9517` | `0.7021` | `10.833` |
| `20190226` | `783` | `0.3032` | `1.9874` | `0.6188` | `9.050` |
| `20190227` | `915` | `0.0161` | nunca | — | `0.026` |
| `20190228` | `798` | `0.2926` | `1.9636` | `0.5355` | `7.834` |

| canal | `20190225` | `20190226` | `20190227` | `20190228` |
|---|---|---|---|---|
| fusões | `226` | `217` | `85` | `202` |
| elásticas | `923` | `837` | `359` | `951` |
| fragmentações | `2131` | `2142` | `1002` | `2256` |
| **total de eventos** | `3280` | `3196` | **`1446`** | `3409` |

Agregado sobre o ensemble: `27.1%` elástica, `6.4%` fusão, `66.5%` fragmentação, `11331` eventos.

**A linha de `20190227` NÃO É UMA MEDIÇÃO FÍSICA.** Ela terminou com `|p_final| = nan` e está
excluída da estatística abaixo. A justificativa da exclusão, e o que ficou em aberto, estão no
bloco próprio mais adiante — **leia-o antes de usar qualquer número desta seção**, porque é
precisamente a linha excluída que teria falsificado a predição.

**Veredito por linha da predição vigente (Seção 4.13.6), sobre as três sementes limpas mais a
rerodada reprodutível de `20190227` (`N_final = 785`, `max m = 307.6 m_bar`, isto é `0.3076`):**

| linha da predição | banda | medido | veredito |
|---|---|---|---|
| `N_final` | `[700, 900]` | `774`, `783`, `798`, `785` | **CONFIRMADO**, `4/4` |
| `t_50` | `> 3 t_ff` | `N_live` nunca abaixo de `774` | **CONFIRMADO**, `4/4` |
| `t_runaway` | `[1.2, 2.0] t_ff` | `1.9517`, `1.9874`, `1.9636` | **CONFIRMADO — mas na borda**, ver abaixo |
| `max_i m_i / M_real` | `[0.15, 0.60]` | `0.3213`, `0.3032`, `0.2926`, `0.3076` | **CONFIRMADO**, `4/4` |
| duração da transição | `> 0.5 t_ff` | `0.7021`, `0.6188`, `0.5355` | **CONFIRMADO — a última a `7%` do piso** |
| `max_t \|E_int\|/\|E_0\|` | `[3, 40]` | `10.833`, `9.050`, `7.834` | **CONFIRMADO**, `3/3` |
| canais, cada `>= 5%` | `[M]` `28.1/6.9/65.0` | `27.1/6.4/66.5`, `11331` eventos | **CONFIRMADO** |
| massa, momento exatos | `[T]` | `3/4` exatos; `1/4` com `\|p\| = nan` | **CONTRADITO em `1/4`** — ver o bloco |

**Critérios de `INV-31`:** **(C1)** nenhuma semente termina com `N_final = 1` — **passa**.
**(C2)** mediana de eventos `>= 50`: mediana `~3238` — **passa com quatro ordens de folga**.
**(C3)** `t_50 > 1.0 t_ff` nas quatro — **passa**. **(C5)** cada canal `>= 5%` agregado: fusão em
`6.4%` é o mais próximo do piso — **passa**. **(C7)** runaway detectado, medido e a ser marcado em
toda figura — **honrado**. **(C6)** — **NÃO REPORTADO**: `min_i m_i / m_bar` não consta dos números
recebidos, e `INV-31(C6)` exige `>= 1e-3` em pelo menos `3` das `4` sementes. **É um critério do
ensemble que ficou por verificar, e não pode ser dado por passado.**

**O que o resultado apoia, dito com o cuidado que ele merece.** O runaway é **robusto às escolhas
estocásticas do modelo de colisão**: três sementes independentes (quatro, com a rerodada) produzem-no,
e produzem-no com dispersão notavelmente pequena — `max m / M_real = 0.306 ± 0.012`, isto é
`CV ≈ 4%` sobre um processo de crescimento descontrolado com `~3300` eventos aleatórios cada.
A decisão da Seção 4.13.5 de aceitar o runaway como resultado **está apoiada** por esta medição, e
deixa de repousar sobre uma única execução.

**Duas observações que qualificam o "bate", e nenhuma delas é decorativa:**

1. **`t_runaway` passa encostado no teto da banda.** Medidos `1.95`, `1.96`, `1.99` contra um teto
   de `2.0` — margens de `2.4%`, `1.8%` e `0.6%`. Uma predição que passa colada na parede é
   evidência mais fraca do que uma que passa no meio: bastaria a banda ter sido `[1.2, 1.9]` para
   as três reprovarem. **E a origem da banda é suspeita** — ver o bloco de divergência sobre
   `t_runaway` logo abaixo.
2. **A dispersão pequena é informação, e aponta para fora do ensemble.** Se o valor final fosse
   fixado pelos sorteios de canal, quatro sementes independentes dariam espalhamento largo. Deram
   `CV ≈ 4%`. **A leitura natural é que o valor final é fixado pela massa disponível na vizinhança
   do corpo em crescimento — que é uma propriedade da realização de POSIÇÕES, e essa foi mantida
   fixa nas quatro execuções.** **[A]** Isso torna a variação da semente de posições **mais**
   importante, não menos, e é o assunto do bloco seguinte.

##### DIVERGÊNCIA: `t_runaway` e a duração da transição não batem com o que este documento já registra

**Encontrada ao confrontar o ensemble com a Seção 4.13.4. Não é resolvida aqui; é uma medição
pendente, e ela toca um argumento que carregou uma decisão.**

A semente `20190225` do ensemble **é a mesma execução** já relatada na Seção 4.13.4 — os três
observáveis independentes batem **até o último dígito publicado**:

| grandeza | Seção 4.13.4 | ensemble 4.13.7 | confere? |
|---|---|---|---|
| `N_final` | `774` | `774` | sim |
| `max m / M_real` | `0.3213` | `0.3213` | sim |
| `max \|E_int\|/\|E_0\|` | `10.83` | `10.833` | sim |
| **`t_runaway / t_ff`** | **`~1.55`** | **`1.9517`** | **NÃO — `26%`** |
| **duração da transição** | **`1.43 t_ff` (`48%` da execução)** | **`0.7021 t_ff` (`23%`)** | **NÃO — fator `2`** |

**A trajetória é a mesma; logo a divergência está na DEFINIÇÃO ou na MEDIÇÃO dessas duas grandezas,
não na física.** **[T]** Duas causas prováveis, e as duas são defeitos deste documento:

1. **`t_runaway ≈ 1.55` pode ser uma sobrevivência da execução DEFEITUOSA de 4.13.2.** Aquela seção
   diz: *"crescimento suave e comportado até `t ≈ 1.55 t_ff` com `max m ≈ 7 m_bar`, e então explosão
   em `~150` passos"*. O `1.55` da constante `ENS_RUNAWAY_TFF` ("medido `~1.55`") pode ter sido
   transportado da rodada 1 para os parâmetros da revisão (c) sem ser remedido na rodada 3. **Seria a
   décima oitava sobrevivência do mesmo tipo**, e a suspeita é reforçada pelo fato de o ensemble dar
   `1.95`–`1.99` nas três sementes limpas, **encostado no teto `2.0` da banda** — que é exatamente o
   que se espera de uma banda centrada num valor que pertence a outra execução.
2. **"duração da transição" NUNCA FOI DEFINIDA OPERACIONALMENTE neste documento.** A Seção 4.13.6
   pede `duracao da transicao > 0.5 t_ff` sem dizer entre quais dois instantes se mede. `t_runaway`
   tem definição (`primeiro t com max_i m_i / M_real >= 0.10`, `INV-31(C7)`); a duração não tem
   nenhuma. **Uma grandeza sem definição operacional dentro de uma predição falsificável é um defeito
   de especificação**, e é a explicação mais econômica para dois medidores obterem `1.43` e `0.70`
   da mesma trajetória.

**O que isto atinge, e é preciso ser franco sobre o tamanho.** A decisão de **aceitar** o runaway
(Seções 4.13.5, 9.7, item 46 da Seção 10) apoia-se, entre cinco razões, numa razão de **produto**:
*"a transição toma `48%` da execução e lê como processo, não como glitch"*, em contraste com `1.2%`
na rodada 1. Se a duração correta for `0.70 t_ff`, o número certo é **`23%`**, não `48%`.

- **O critério declarado continua sendo satisfeito:** `0.70`, `0.62` e `0.54 t_ff` estão todos acima
  do piso `> 0.5 t_ff`, e `23%` está acima do `17%` que a própria Seção 4.13.6 dá como equivalente.
  **A conclusão não vira.**
- **Mas o argumento é mais fraco do que está escrito**, por um fator `2`, e a semente `20190228`
  passa a `7%` do piso. **Isso é diferença entre "toma metade da execução" e "toma nem um quarto
  dela"**, e a prosa do relatório não pode continuar dizendo `48%` sem que alguém tenha remedido.

**Medição que resolve, e é obrigatória antes de qualquer figura ou prosa sobre a fase colisional:**
fixar a definição operacional da duração da transição — a proposta natural, coerente com `(C7)`, é
`t(max m/M_real = 0.60 * valor final) - t(max m/M_real = 0.10)` sobre a grade **por passo** — e
recomputar `t_runaway` e a duração para `20190225` com as definições escritas. Até lá, **`1.55` e
`48%` estão marcados como NÃO CONFIRMADOS** e não podem ser citados.

##### O que o ensemble testa, e o que ele NÃO testa

**Este é o enunciado correto da ressalva que uma versão anterior deste documento formulou como a
afirmação errada de que o ensemble não tinha rodado.**

O protocolo da Seção 4.13.6 fixa a semente de posições e varia só a de colisão. Portanto as quatro
execuções compartilham **a mesma realização de Poisson da esfera**: as mesmas flutuações de
densidade, a mesma estrutura de núcleo, os mesmos vizinhos próximos, o mesmo conjunto de partículas
que termina no núcleo.

| eixo de variação | testado? | o que se conclui |
|---|---|---|
| sorteios de canal do mapa de regime (`u1`, `u2`) | **sim**, `4` sementes | o runaway não é artefato de uma sequência aleatória particular |
| realização da condição inicial (posições) | **NÃO** | nada |

**Por que isto importa neste modelo especificamente, e não é uma ressalva genérica.** O runaway é um
fenômeno de **núcleo**: a taxa de encontros escala com `n²` da densidade local, e qual corpo "sai na
frente" depende de quem está mais densamente cercado. Ambas são propriedades da realização de
posições. O ensemble atual mede a robustez **condicionada** a essa realização.

**Ordem de grandeza do que se esperaria variando as posições** **[A]**, e é modesta: com `N = 1000`
e `~500` corpos no núcleo, o ruído de Poisson na densidade de núcleo é `~1/sqrt(500) ≈ 4.5%`, e a
taxa de encontros varia com o quadrado, `~9%`. **Isso é uma estimativa, não uma medição**, e é
exatamente do mesmo tipo de argumento de campo médio que já falhou três vezes neste documento
(Seções 4.12, 9.7) — **razão pela qual ela não pode substituir a medição**.

**A medição que fecha o assunto, e é barata.** Repetir o ensemble variando **`config.SEED`** (a
semente de posições) e mantendo a de colisão fixa, com `K = 4`. Critério de leitura, registrado
**antes** de executar, para que possa falhar:

```
max_i m_i / M_real em 3 t_ff, sobre 4 realizacoes de posicoes:
  todas em [0.15, 0.60]          -> o runaway e' propriedade do colapso frio a chi = 0.1   [A]
  alguma abaixo de 0.05          -> o runaway depende da realizacao; a Sec. 4.13.5 decidiu
                                    com um ensemble condicionado e precisa ser reaberta     [A]
  dispersao >> CV = 4% observado -> o valor final e' fixado pela IC, como 4.13.7 sugere;
                                    entao TODA figura precisa declarar de qual realizacao fala
```

**Enquanto essa medição não existir, a redação obrigatória é esta**, e vale para o relatório e para
toda figura: *"o runaway é robusto às escolhas estocásticas do modelo de colisão, medido sobre `4`
sementes de colisão numa realização fixa da condição inicial"*. **É proibido escrever** *"o runaway
é robusto entre realizações"* ou *"o colapso frio a `chi = 0.1` produz runaway"* — a segunda
generaliza de uma realização para o fenômeno, e é a mesma forma de erro que a Seção 4.7 já proíbe
para as frações de canal ("é proibido escrever 'o colapso produz `X%` de fusões'").

##### A execução que não reproduziu — `20190227`, `|p_final| = nan`

**Registrada aqui em vez de escondida ou promovida a defeito conhecido. É um evento único, não
reproduzido, e a causa não foi identificada.**

**O que aconteceu.** Na campanha, `20190227` terminou com `|p_final| = nan` e com números anômalos
em todas as grandezas: `N_final = 915`, `max m / M_real = 0.0161`, `|E_int|/|E_0| = 0.026`, e
**`1446` eventos contra `~3300` das outras três**.

**O que foi verificado, e por quem investigou o incidente:**

- a mesma semente foi reexecutada **três vezes** pelo mesmo caminho de código — uma isolada, duas
  dentro de um laço — e deu `N_final = 785`, `max m = 307.6 m_bar` **nas três, sem `NaN`**;
- as outras três sementes reproduzem **até o último dígito impresso** entre campanha e rerodada;
- `accelerations`, `detect` e `pair_disjoint` foram verificadas **bit a bit determinísticas** em
  `200` repetições cada, com candidatos reais.

**Diagnóstico do que os números anômalos significam — e eles são consistentes com uma única causa.**
`NaN` propaga para `v`, daí para `r`. **Toda comparação com `NaN` é falsa**, logo `detect` deixa de
emitir candidatos e o maquinário de colisão **congela**. Isso explica as quatro anomalias de uma vez:
os eventos param (`1446` em vez de `~3300`), a massa máxima para de crescer (`0.0161`), `E_int` para
de acumular (`0.026`), e `N_final` fica alto (`915`, com exatamente `85` fusões — a contabilidade de
slots continua **exata**). **[T]**

**Consequência que decide o uso da linha:** esta execução **não mediu uma semente sem runaway**. Ela
mediu uma execução que **parou de simular** por volta de metade do caminho. Uma semente genuinamente
sem runaway teria contagem de eventos **normal** com massa distribuída — não `44%` dos eventos. Logo
ela **não é evidência de bimodalidade**, e a cláusula de reabertura da Seção 4.13.6
(`max m/M_real < 0.05 -> reabrir`) **não é acionada por ela**.

> **A coincidência precisa ser dita em voz alta, porque é exatamente o padrão que este documento
> existe para impedir.** A única das quatro execuções que teria falsificado a predição vigente é
> a que foi excluída. Um leitor cético tem o direito de suspeitar de descarte conveniente, e a
> resposta não pode ser "confie". A resposta é que a exclusão se apoia em **três coisas verificáveis
> e nenhuma delas é o valor inconveniente**:
>
> 1. **Uma regra pré-declarada.** A Seção 4.13.6 já listava, antes desta execução existir, a linha
>    *"massa ou momento inexatos -> defeito de implementação — nada nesta seção o autoriza"*.
>    `|p| = nan` é momento inexato. **A regra que exclui esta execução foi escrita antes de ela
>    ocorrer**, e exclui-a por `NaN`, não por `0.0161`.
> 2. **A mesma semente, reexecutada três vezes, produz runaway** (`0.3076`), dentro da banda e
>    dentro do `CV = 4%` das outras. O valor `0.0161` não é o que a semente produz.
> 3. **Um mecanismo que explica as quatro anomalias simultaneamente**, e prevê corretamente a
>    contagem de eventos reduzida — que não é um número que se ajustaria a posteriori.
>
> **Se qualquer uma das três caísse, a exclusão cairia junto.** Em particular: se uma rerodada
> produzisse `0.0161` sem `NaN`, a cláusula de reabertura estaria acionada e a Seção 4.13.5 teria
> de ser reaberta.

**O que fica em ABERTO, e não deve ser fechado por conveniência.** Tudo neste código é uma função
determinística de `(semente, código, aritmética)`. **Um resultado que não reproduz sob a mesma
semente e o mesmo código significa que algo fora dessa tripla variou.** Consequências:

- **toda hipótese de caminho de código determinístico está excluída como causa única** — inclusive a
  candidata mais natural, o caso degenerado `|sep| = 0` da Seção 4.9(0.1), que produziria `NaN` em
  `v` mas o produziria **sempre** para a mesma semente;
- **resta**: (i) ordem de redução variável em `accelerations` (o `tile_size` é escolhido a partir de
  um teto de memória, e a campanha de `4` sementes num processo tem pressão de memória diferente de
  uma execução isolada) — **enfraquecida** pelo fato de as outras três reproduzirem bit a bit, o que
  indica que a ordem de redução foi a mesma; ou (ii) uma falha transitória de hardware/memória, que
  é inverificável após o fato e é o resíduo quando (i) cai.

**Instrumentação recomendada, barata, para o caso de recorrer** (não implementada; não é pedido de
mudança de código agora): asserção de finitude sobre `v` a cada `N` passos no laço de integração,
com **despejo do índice do passo e do estado no primeiro `NaN`**. Sem o índice do passo, uma
recorrência é tão pouco informativa quanto esta.

**Perigo latente, INDEPENDENTE deste incidente, que a investigação fez aparecer e que deve ser
registrado.** A regra normativa de 4.9(0.1) guarda o caso `sep_sq == 0.0` com uma comparação
**exata**. Ela não guarda o caso `sep_sq` **muito pequeno e não nulo**: com `sep_sq ~ 1e-300`,
`sqrt(sep_sq) ~ 1e-150` e `n = sep / 1e-150` pode **estourar para `inf`**, e `u' = u - 2(u.n)n`
produz então `inf - inf = NaN`. **É uma descontinuidade de guarda protegendo um modo de falha
contínuo.** **[T]** Isto **não** explica o incidente (seria determinístico), e **não** é uma
mudança que este documento autoriza fazer agora — é um apontamento para a revisão de `resolve`, com
a correção óbvia sendo comparar contra um piso relativo em vez de contra zero exato.

### 4.14 O PISO — os oito pontos sem liberdade

Esta subseção existe porque o restante do documento foi deliberadamente afrouxado (Seção 9.5), e o
implementador precisa saber **onde o afrouxamento para**. Fora desta lista, uma aproximação
grosseira, uma constante escolhida a dedo ou um modelo fenomenológico são escolhas legítimas de
projeto. Dentro dela, não há escolha.

> **O item 6 foi RETIRADO em 2026-08-07 (b), e a numeração dos demais está PRESERVADA** para não
> invalidar referências. Eram nove; são oito. Ver a linha `~~6~~` abaixo, e a Seção 4.6.1.
>
> **Uma lição sobre esta lista, que vale mais que o item retirado.** O item 6 entrou no piso
> justificado por um *defeito visível* ("sem o termo gravitacional nada nunca funde"), não por uma
> conservação exata. Ele era o **único** item do piso nessa condição — e foi o único que caiu.
> Os outros oito protegem conservações exatas ou finitude numérica, e nenhum deles dependeu de uma
> premissa dinâmica que uma medição pudesse refutar. **Um item de piso que precise de um argumento
> sobre o comportamento do sistema para se sustentar não é um piso; é uma previsão.** Novos itens só
> entram aqui se quebrarem uma conservação, produzirem `NaN`, ou destruírem a reprodutibilidade.

Cada item traz **o que quebra** se alguém o simplificar depois. Nenhum destes é uma questão de
precisão: são conservações exatas, ou defeitos visíveis na tela.

| # | O que é obrigatório | O que quebra se sair |
|---|---|---|
| **1** | O impulso elástico é aplicado **em `t*`**, com `n` paralela à separação **naquele ponto** (Seção 4.5, 4.9) | `L` deixa de conservar. E `K` continua passando, porque a reflexão conserva `K` para qualquer `n` unitária — **nenhum outro teste pega isso**. É o par de testes mais discriminante da suíte (`INV-20`). |
| **2** | O corpo fundido vai para o **centro de massa do par** (Seção 4.9) | `sum_i m_i r_i` salta a cada fusão. Com `P = 0`, o centro de massa do sistema deveria ficar parado; ele passa a **derivar visivelmente na tela**. |
| **3** | `m_b = M - m_a`, e a partição de massa soma `M` (Seção 4.9) | A massa total não fecha. `INV-19(a)`. |
| **4** | O pareamento é **disjunto**: um evento por slot por passe (Seção 4.5) | Um slot é consumido por dois mapas e massa e momento se perdem. É combinatório, não numérico: não há tolerância que o salve. |
| **5** | **Nunca** reposicionar corpos sobrepostos "até se tocarem" (Seção 4.5, veto) | Afastar corpos torna `U` menos negativo: **injeta energia mecânica do nada**. O aglomerado aquece e se desliga. No esquema adotado a sobreposição nunca se forma, então não há o que corrigir. |
| ~~**6**~~ | ~~`E_bind` mantém o termo gravitacional~~ **RETIRADO 2026-08-07 (b)** | A justificativa (`x >= 1` identicamente sem ele) era **falsa**: o piso real era `~0.4`, não `1` (Seção 4.6.1). E o termo **causava** o crescimento descontrolado que se acreditava que ele contivesse. Medido: `max m_i = 321 m_bar` (Seção 4.13.2). `v_esc_eff` não existe mais. |
| **7** | Se `E_int` existir, o **termo mútuo** `E_grav(m_i,m_j,d_ij)` está dentro dela (Seção 4.10) | `~10%` de `\|E_0\|` de dissipação fabricada, de sinal único, ao longo de uma execução. **[M]** Uma curva de energia visivelmente indo embora no HUD. |
| **8** | `eps > 0` com colisões ligadas; `ValueError` se `softening == 0.0` (Seção 5.2, `INV-28`) | `0 * inf -> NaN` no campo de aceleração no primeiro slot morto coincidente com um corpo vivo. **A simulação inteira vira `NaN`.** |
| **9** | O desempate `(t*, i, j)` na ordenação do pareamento (Seção 4.5) | Não é física: é reprodutibilidade. Duas execuções da mesma semente divergem, e entre dispositivos a execução deixa de ser publicável. `INV-19(c)`. |

**Regra de leitura para o implementador.** Se uma simplificação proposta não atinge nenhum destes
oito pontos, ela provavelmente é aceitável e a decisão é sua. Se atinge, ela não é. Em caso de
dúvida sobre se atinge, ela atinge.

**Nota sobre o item 7.** Ele é condicional (*"se `E_int` existir"*) de propósito. Cortar `E_int`
inteiro e rotular a curva do HUD como `E_mec = K + U`, com os degraus de colisão visíveis, é uma
saída **permitida** e honesta — só não se pode chamá-la de `E_total` nem afirmar conservação. O que
é proibido é a versão intermediária: manter `E_int` e omitir o termo mútuo dela.

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

### 5.3 `half_mass_radius` — a correção está FEITA

> **Estado em 2026-08-07: IMPLEMENTADA E COMMITADA.** `src/nbody/observables.py:70` já usa
> `argsort` + `cumsum` + `searchsorted`, isto é, a mediana de **massa** especificada abaixo.
> **Não há trabalho pendente aqui**, e a Seção 9.3 registra o item como fechado. O texto abaixo fica
> como justificativa da mudança e como enunciado normativo contra o qual `INV-29` testa.

**Confirmação do apontamento original.** A versão anterior de `src/nbody/observables.py` devolvia
`sorted_d[n // 2 - 1]`: a mediana de **contagem**. Isso é a mediana de **massa** apenas quando as
massas são iguais **e** `N` é par. Verificado **[M]**:

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

### `INV-18` — Álgebra da detecção varrida e ausência de perda por tunelamento

**Enunciado (a).** `t*` é o minimizador de `|dr + t dv|²` em `[0, h]`.

**Procedimento (a).** `200` configurações aleatórias, incluindo casos que ativam cada extremo do
clamp e o caso `|dv|² = 0`. Comparar com minimização por varredura de `1e5` pontos.

**Tolerância (a).** `|t*_formula - t*_grade| <= h/1e5` e
`| |sep|_formula / |sep|_grade - 1 | <= 1e-12`. Justificativa: o erro da grade é `O(h/1e5)` por
construção; a segunda cota é de arredondamento porque a parábola é plana no mínimo.

> **REVISÃO 2026-08-07: a cláusula (b) mudou de natureza.** A versão anterior exigia
> `max C_coll <= 1` "sem margem", como condição de validade, e mandava reduzir `dt` se falhasse.
> A Seção 4.4 revisada demonstra **[T]** e mede **[M]** que essa condição não existe.

**Enunciado (b), Courant colisional — REPORTADO, NÃO BLOQUEANTE.**
`C_coll = |u| dt / (R_i+R_j)` é calculado sobre todos os candidatos e o seu máximo é gravado no CSV
de cada execução e exibido no HUD. **Nenhum teste falha por causa do seu valor**, e nenhuma execução
é invalidada por ele. Valor de referência a `chi = 0.1`, `dt = 5.0e-4`: `c_coll_max = 1.81` **[M]**.

**Procedimento (b).** Verificar que a grandeza **é exposta** e que é numericamente igual a
`rel_speed * dt / contact_radius_sum` sobre os candidatos. Isto testa a contabilidade, não um limiar.

**Enunciado (c), NOVO — o detector varrido não perde contatos.** Sobre uma reta, `dr . dv` é
monotonicamente não decrescente, logo o passo que contém a aproximação máxima tem `dr . dv < 0` no
seu início e o `clamp` devolve o mínimo interior exato (Seção 4.4.2). **[T]**

**Procedimento (c).** Geometria pura, sem chamar o detector real. Para cada
`C_coll ∈ {0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 20.0}` e para parâmetro de impacto uniforme no disco
de raio `R` **e** para o caso frontal (`b = 0.01 R`), gerar `>= 2000` encontros com fase de grade
aleatória, aplicar a lógica exata do detector passo a passo, e exigir que **todo** encontro cujo
mínimo verdadeiro está dentro de `R` seja detectado em algum passo.

**Tolerância (c).** Fração perdida **exatamente `0`**. Binária, sem margem: é uma propriedade
estrutural, não estatística. Medido `0.000000` em `4.0e6` encontros. **[M]**

**Se falhar (c).** A guarda de aproximação está sendo aplicada fora do início do passo, ou o `clamp`
de `t*` está errado, ou a detecção está usando a configuração de fim de passo. É o teste que pega a
regressão para o esquema que a Seção 4.5 proíbe.

### `INV-19` — Pareamento disjunto: massa, determinismo, e taxa de rejeição

**Enunciado (a).** `|sum_i m_i (depois) / sum_i m_i (antes) - 1| <= n_events * eps_prec`, por passe.
Derivada de um arredondamento por evento (Seções 4.5 e 4.9).

**Enunciado (b).** Nenhum slot participa de dois eventos no mesmo passe. Teste direto sobre a lista
de eventos aceitos.

**Enunciado (c).** Determinismo: duas execuções com as mesmas sementes produzem listas de eventos
idênticas, na mesma ordem, incluindo em caso de `t*` empatado. Construir deliberadamente um caso
com dois pares de `t*` idêntico bit a bit (configuração simétrica) e verificar reprodutibilidade.

**Enunciado (d), REVISADO — `f_reject` é POR PASSE, e é reportado.**
`AcceptedPairs.f_reject` é `n_rejeitados / n_candidatos` **do passe** (Seção 4.5), e vale `0.0` num
passe sem candidatos. **Testável de forma determinística e sem execução completa:** construir um
conjunto de candidatos com conflito conhecido (por exemplo `4` candidatos dos quais `2` são
rejeitados por disjunção) e exigir `f_reject == 2/4` exatamente.

Os agregados `f_reject_max` e `f_reject_total` são responsabilidade do laço de integração, não de
`pair_disjoint`, e são **reportados** no CSV. **Nenhum teste falha por causa do seu valor.** A linha
de atenção de `0.05` (Seção 4.5) é editorial: acima dela o relatório declara que a sequência de
eventos é aproximada. Referência medida a `chi = 0.1`, `dt = 5.0e-4`: `f_reject_total = 0.00407`
**[M]**.

> A versão anterior exigia `f_reject <= 0.05` "sobre toda a execução" sem definir o agregado, e sem
> um caminho para medi-lo antes de `resolve()` existir — o que tornava `INV-19(d)` intestável. A
> redefinição por passe o torna testável **hoje**, com um conjunto de candidatos sintético.

**Se falhar (c).** Chave de ordenação sem desempate por `(i, j)` — o resultado deixa de ser
reprodutível entre dispositivos e a execução não é publicável.

**Se falhar (d).** A contabilidade de rejeição está errada (numerador ou denominador trocados, ou
`f_reject` acumulado entre passes em vez de por passe). Não é um sintoma de `dt` grande demais.

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

> **REVISÃO 2026-08-07: a cláusula de ISOTROPIA de `INV-22` foi REMOVIDA.** Ela dizia: *"`5000`
> eventos sintéticos; o tensor de segundo momento de `u'/|u'|` tem autovalores em `[0.30, 0.37]`."*
> A direção isotrópica sorteada deixou de existir (Seção 4.9): `u'` é agora **alinhado com a normal
> de contato `n`**, de modo que a cláusula não tem mais objeto — testar isotropia de uma direção
> deliberadamente não isotrópica reprovaria uma implementação correta.
>
> **`INV-22` permanece em vigor**, com todas as suas outras cláusulas inalteradas. Só a cláusula de
> isotropia saiu, e foi substituída pela de alinhamento abaixo.

**Alinhamento com a normal de contato (substitui a isotropia).** Em todo evento de fragmentação,
`u'` é paralelo a `n` e aponta no sentido da separação:

```
|  u' . n / |u'|  -  1  |  <=  100 eps_prec              (paralelo a +n, nao a -n)
|  (r_a - r_b) . n / |r_a - r_b|  -  1  |  <=  100 eps_prec
|r_a - r_b| / (R_a + R_b)  -  1     em modulo  <=  100 eps_prec     (postos EM CONTATO)
```

A primeira cota é o teste discriminante: um sinal trocado em `u'` põe os fragmentos a se
aproximarem, o que produz recolisão imediata e um moinho de eventos no passo seguinte.

**Razão de massa.** `f ∈ [0.1, 0.9]` em todos os eventos (determinístico), e a distribuição de
`f` é uniforme por KS (`p >= 0.01`). Com `f = 0.1 + 0.8 * u2` e `u2 ~ U(0,1)` (Seção 4.7.1), isto
é exato por construção.

### `INV-23` — `E_total = K + U + E_int` e o sinal de `E_int`

**Enunciado (a), por evento — AFROUXADO nesta revisão.** Recalculando `K + U + E_int` imediatamente
antes e depois de cada mapa, com posições congeladas:

```
| Delta(K + U + E_int) | / |E_0|   <=   TOL-EVENT-CONS  =  1e-5     (fp64)
```

> **De `1e-13` para `1e-5`, e a razão não é tolerância numérica.** `E_int` é agora computada em forma
> fechada ao nível do par (Seção 4.10), **omitindo deliberadamente os termos de terceiro corpo**.
> O resíduo por evento foi medido: `2.5e-6 |E_0|`, isto é `1.2%` do termo mútuo, de sinal único
> **[M]**. Uma cota de `1e-13` reprovaria uma implementação **correta** pela omissão que este
> documento autoriza. `1e-5` acomoda o resíduo medido com margem `4x` e continua **oito ordens**
> abaixo do degrau de um evento (`2e-4 |E_0|`), de modo que o teste **continua pegando** termo
> esquecido ou sinal trocado no acumulador — que é o que ele existe para pegar.
>
> **Cota inferior obrigatória, para o teste não virar vazio:** exigir também
> `max_evento |Δ(K+U+E_int)| / |E_0| >= 1e-8`. Se o resíduo for do nível de arredondamento, os termos
> de terceiro corpo **estão** sendo computados, contra a Seção 4.10 — o que não é um erro de física,
> mas é uma divergência entre código e especificação, e é exatamente o tipo de coisa que este
> documento existe para impedir.

**Deriva acumulada esperada, e NÃO é falha.** Ao longo de `3 t_ff` com `~500` eventos, o resíduo de
sinal único acumula `~1.2e-3 |E_0|`, isto é **`0.12%`** **[M]**. Isso é o comportamento **previsto**
de uma implementação correta sob esta especificação. Um teste que exija `|ΔE_total/E_0| < 1e-3` numa
execução com fusão reprova código correto.

> # EMENDA 2026-08-08 (d) — RECONCILIAÇÃO COM AS SEÇÕES 4.10 E 7
>
> **Esta seção exibia, até esta emenda, o texto RETRATADO de `(c)`** (`min_t E_int >= -1e-2 |E_0|`)
> e um bloco "Se (c) falhar" que mandava não afrouxar a cota — depois de a Seção 4.10 (2026-08-08)
> e a Seção 7 já terem retirado `(c)` da condição de cota. **Um invariante que o próprio documento
> revoga noutra seção é pior que invariante nenhum: alguém escreve teste contra ele.** Reconciliado
> abaixo. Nada de novo foi decidido aqui — esta emenda apenas propaga para a Seção 6 a decisão já
> tomada e justificada em 4.10, 4.13.5 e 9.7.
>
> **Enunciados `(b)` e `(a)` também ganharam regime de validade declarado**, pela mesma razão: as
> duas cotas foram derivadas de medições do estágio 2 (pares de massa igual, sem corpo dominante) e
> a execução aceita da Seção 4.13.4 sai desse regime a partir de `t_runaway ≈ 1.55 t_ff`. Isso é
> **escopo**, não afrouxamento: a cota continua valendo, sem alteração de valor, onde a sua
> derivação vale. O escopo é o que a própria Seção 4.13.5 já exigia declarar em toda figura e todo
> texto sobre a fase colisional.

**Enunciado (b), ao longo da execução — VÁLIDO SÓ ANTES DE `t_runaway`.** Enquanto
`max_i m_i / M_real < 0.10`, `|ΔE_total/E_total(0)|` obedece aos critérios **qualitativos** de
`INV-4`, transferidos de `E_mec` para `E_total`: `velocity_verlet` não monótono e com final
`<= pico/10`; `euler` monótono crescente com final `>= +0.3`; `rk4` com final negativo e
`>= pico/3`. **Os valores `[M]` de `integradores.md` não se transferem** — a trajetória é outra.

> **Por que o escopo é obrigatório, e não uma conveniência.** A Seção 4.10 (2026-08-08) proíbe
> explicitamente afirmar qualquer coisa sobre qualidade de integração a partir de `E_total` quando
> `|E_int| > |E_0|`. Medido `max_t |E_int|/|E_0| = 10.83` **[M]**: nesse regime `E_total` é a soma
> de três parcelas de ordem `10 |E_0|` cujo cancelamento é dominado pelo livro de colisões, e a
> forma da curva deixa de ser propriedade do integrador. Aplicar `(b)` ali **testa o modelo de
> colisão acreditando testar o integrador**. Fora do regime colisional, `INV-4` de
> `integradores.md` continua em vigor sem emenda e é lá que os integradores são comparados
> (Seção 4.11, proibições explícitas).

**Enunciado (c) — RETIRADO DA CONDIÇÃO DE COTA em 2026-08-08. `E_int` é REPORTADA, não vigiada.**

```
max_t |E_int(t)| / |E_0|      REPORTADO com sinal, exibido no HUD e na figura.   [M] 10.83
                              NENHUM teste falha por causa deste valor.
```

Três formulações anteriores, todas cotas, todas reprovadas — preservadas só como história:
`min_t E_int >= -1e-3 |E_0|`, depois `>= -1e-2 |E_0|`, depois `max_t |E_int| <= 1.0 |E_0|`.
Elevar a cota uma quarta vez seria o ajuste post-hoc que este documento proíbe; o que se fez foi
mudar o **tipo** do critério por argumento (Seções 4.10, 4.13.5, 9.7). `|E_int| >> |E_0|` é a
assinatura quantitativa do runaway aceito, e um número que mede fielmente um resultado aceito não
pode reprovar esse resultado.

**Rótulo obrigatório sempre que `|E_int| > |E_0|`:** o livro de colisões movimentou mais energia do
que existe ligando o aglomerado, e a curva de `E_total` **deixa de ser um diagnóstico do
integrador**.

**O que substitui `(c)` como guarda, e continua bloqueante:** `(a)`, o resíduo por evento, que é o
teste que pega sinal trocado ou termo esquecido no acumulador (Seção 4.10), e `INV-20/21/22`, que
testam a física de cada desfecho. `(c)` nunca foi capaz de pegar nenhum dos dois — as três
reprovações foram do fenômeno, não da implementação.

**Enunciado (a) — regime de validade da normalização.** O teto `1e-5` e o piso `1e-8` estão
normalizados por `|E_0|` e foram derivados do resíduo de terceiros medido em pares de massa igual
no pico de compressão (`2.5e-6 |E_0|`, `1.2%` do termo mútuo **[M]**). O resíduo escala com o
**termo mútuo do evento**, isto é, com `m_i m_j`; um evento envolvendo o corpo de `321 m_bar` tem
termo mútuo `~0.03 |E_0|` e portanto resíduo esperado `~4e-4 |E_0|`, **`40x` acima do teto**. Logo:

- **antes de `t_runaway` (`max_i m_i / M_real < 0.10`): a cota `1e-5` vale como escrita**, e é
  bloqueante;
- **depois: a normalização por `|E_0|` perdeu sentido** e a forma correta da cota é relativa ao
  termo mútuo do próprio evento (`|Δ(K+U+E_int)| / E_grav(m_i,m_j,d_ij) <= 5e-2`, que é `4x` o
  `1.2%` medido). **Essa forma NÃO foi medida** e está marcada **[A]**; até que seja, o resíduo
  por evento no regime pós-runaway é **reportado**, não vigiado. A medição que a converte em
  **[M]** é um histograma de `|Δ(K+U+E_int)| / E_grav_mutuo` sobre os `3280` eventos da execução
  da Seção 4.13.4.

**Se (a) falhar pelo teto, no regime em que ela vale.** Sinal trocado no acumulador, ou o termo
mútuo `E_grav` ausente de `E_int`. **Se (a) falhar pelo piso** (resíduo abaixo de `1e-8`), os
termos de terceiro corpo estão sendo computados contra a Seção 4.10.

### `INV-24` — `L_total = L_orb + L_spin`

**Enunciado.** `|ΔL_total| / L_SCALE <= 1e-12` em fp64 ao longo de `RUN_COLLISION` com
`velocity_verlet`, com `L_SCALE = M_real * R_0 * V_CHAR` recalculado da massa realizada.

**Enunciado complementar, de poder discriminante.** `|L_spin(t_end)| / L_SCALE` deve ser **não
desprezível** — cota inferior `1e-8`. Se `L_spin` permanecer no nível de arredondamento, ou os
eventos não estão ocorrendo (então `INV-31(C2)` também falha), ou o termo de spin não está sendo
acumulado, e `INV-24` estaria passando vazio.

### `INV-25` — O mapa de regime é bem formado

**Reescrito nesta revisão** para o mapa `(1/x, 3, x)/Z` da Seção 4.7. As cláusulas que dependiam de
`w` foram retiradas porque `w` não existe mais; a cláusula de simetria é nova e não tinha equivalente.

**Enunciado.** Sobre `x ∈ [1e-14, 1e14]` em `601` pontos logarítmicos:

1. **Soma.** `|p_fus + p_el + p_frag - 1| <= 4 eps_prec`;
2. **Positividade estrita.** `p_c > 0` para todo `c` e todo `x`, **inclusive nos clamps**;
3. **Monotonicidade.** `p_fus` estritamente decrescente e `p_frag` estritamente crescente em `x`
   (dentro de `1e-15` por passo, para absorver arredondamento);
4. **Máximo elástico.** `p_el` máxima em `x = 1`, com `|p_el(1) - 0.6| <= 4 eps_prec` — valor exato
   `3/5`;
5. **Simetria (NOVA).** `|p_fus(x) - p_frag(1/x)| <= 100 eps_prec` e
   `|p_el(x) - p_el(1/x)| <= 100 eps_prec`, para todo `x` da grade. Esta é uma identidade algébrica
   exata do mapa e um teste barato e forte;
6. **Travessias em forma fechada.** `p_fus(1/3) = p_el(1/3)` e `p_frag(3) = p_el(3)`, dentro de
   `100 eps_prec`;
7. **Extremos.** Em `x = 1e-300` e `x = 1e300` (isto é, além do `clamp`), `min_c p_c >= 1e-25` —
   previsto `X_CLAMP^-2 = 1e-24` **[T]**, acima do menor normal de fp32 (`1.18e-38`) com `14` ordens
   de folga. Verificar também ausência de `inf`/`nan` em `Z`.
8. **Controle uniforme.** Quando o mapa é substituído pelo controle (Seção 4.7), devolve
   `(1/3, 1/3, 1/3)` dentro de `4 eps_prec`. Isto é uma **escolha explícita** da implementação, não um
   valor limite de parâmetro.

**Tolerâncias.** Todas de arredondamento ou estruturais; nenhuma ajustada a saída.

**Se (2) ou (7) falhar.** `clamp` ausente ou `X_CLAMP` grande demais. Com `X_CLAMP = 1e30` a menor
probabilidade seria `1e-60`, que **subnormaliza a `0.0` em fp32** e viola o requisito explícito do
projeto de que nenhum canal tenha probabilidade exatamente zero.

**Se (5) falhar.** Numeradores trocados entre `p_fus` e `p_frag`, ou `1/x` computado como `x` em
algum ramo. É o teste mais barato da lista e o que pega a inversão de canal.

### `INV-26` — Consistência entre o mapa e o sorteador

**Enunciado.** As frações de canal realizadas concordam com as previstas pela integração do mapa
sobre o histograma de `x` efetivamente visitado:

```
| f_c - <p_c> |  <=  3 * sqrt( <p_c (1 - p_c)> / n_events )
```

com `<·>` a média sobre os eventos registrados. Esta é a cota binomial a `3` desvios-padrão, derivada
e não ajustada.

**Procedimento.** Registro de eventos do estágio 3, agregado sobre as `4` sementes (Seção 4.13).

**Critério adicional (calibração).** Cada `f_c >= 0.05`. Se falhar **pelo lado da fusão**, a única
alavanca autorizada é elevar `v_coh` acima de `V_CHAR` (Seção 4.6) — **antes** de olhar qualquer outro
resultado. O mapa não tem mais parâmetros de forma (`b`, `w` foram removidos), de modo que não há
outra coisa a mexer.

**Se falhar a cota binomial.** O `x` usado no sorteio não é o `x` registrado — tipicamente `E_bind`
computado com o potencial não suavizado num lugar e suavizado no outro.

### `INV-32` — O fluxo de colisão consome exatamente `2` sorteios por evento aceito

**Enunciado.** Após um passe de colisão com `n_events` eventos aceitos, o `Generator` de colisão
consumiu **exatamente `2 * n_events`** valores uniformes, **independentemente dos canais sorteados**.
Um passe com zero eventos consome zero. Seção 4.7.1.

**Procedimento.** Dois caminhos, ambos obrigatórios:

1. **Contagem direta.** Instrumentar (ou clonar) o `Generator`, executar um passe com um número
   conhecido de eventos aceitos, e comparar o número de valores retirados com `2 * n_events`.
2. **Equivalência de fluxo, que é o teste forte.** Construir dois passes com o **mesmo número de
   eventos aceitos** mas **canais diferentes** (por exemplo, forçando `x` muito baixo num caso e
   muito alto no outro), partindo do mesmo estado de `Generator`. Exigir que o estado do `Generator`
   **ao final dos dois passes seja idêntico**. Se `u2` estiver sendo sorteado só dentro do ramo de
   fragmentação, os dois estados divergem e o teste falha.

**Tolerância.** Binária. Sem folga.

**Se falhar.** O `u2` está condicional ao canal, ou existe um terceiro sorteio (candidato mais
provável: uma direção isotrópica remanescente na fragmentação, que a Seção 4.9 removeu).
**Bloqueante:** sem passo fixo, `INV-19(c)` não pode ser satisfeito e nenhuma execução colisional é
reprodutível.

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

> **Estado: a implementação já satisfaz este invariante.** `src/nbody/observables.py:70` usa
> `argsort` + `cumsum` + `searchsorted`. `INV-29` permanece em vigor como teste de regressão — o que
> ele guarda agora é que ninguém volte à mediana de contagem —, mas **não** descreve trabalho
> pendente.

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

> **EXECUTADO em 2026-08-08. Resultados, vereditos e ressalvas na Seção 4.13.7.** **[M]** Resumo:
> **(C1)**, **(C2)**, **(C3)**, **(C5)** e **(C7)** passam; **(C8)** passa (`N_final ∈ {774, 783,
> 798}` mais `785` da rerodada). **(C6) NÃO FOI REPORTADO** — `min_i m_i / m_bar` não consta dos
> números da campanha, e este invariante **não pode ser dado por passado** enquanto ele faltar.
> Uma das quatro execuções (`20190227`) terminou com `|p| = nan`, não reproduziu em três
> reexecuções, e está excluída da física com justificativa pré-declarada — ver 4.13.7.
>
> **Escopo do que foi testado, e é normativo ao citar este invariante:** o protocolo varia **apenas
> a semente de colisão**, com a realização de posições **fixa**. `INV-31` mede robustez às escolhas
> estocásticas do modelo de colisão, **não** robustez entre realizações da condição inicial.

Os critérios (C1), (C2), (C3) e (C5) da Seção 4.13 — **(C4) foi retirado nesta revisão**, ver 4.13 —
sobre **`K_SEEDS = 4`** sementes, mais:

**(C6) Moinho de fragmentação.** `min_i m_i / m_bar >= 1e-3` em pelo menos `3` das `4` sementes, e o
número de corpos abaixo de `m_min` é reportado. **[A]** — a cota `1e-3` é de projeto: abaixo dela o
raio de contato caiu por `10x` e o corpo é efetivamente não colisional, de modo que o moinho se
extingue sozinho; o critério existe para tornar isso visível, não para impedi-lo.

**~~(C8) Predição de `N_final`, registrada antes da implementação.~~ SUPERSEDIDA — ver a `(C8)`
vigente ao fim desta lista.** Enunciado original, preservado como história: `N_final ∈ [700, 800]`,
`t_50 > 3 t_ff` **[A]**, derivação na Seção 4.13.1.

> **Emenda 2026-08-08 (d): esta seção tinha DUAS cláusulas `(C8)`**, esta e a do fim da lista, com
> bandas diferentes (`[700, 800]` contra `[700, 900]` da Seção 4.13.6). Um rótulo duplicado com dois
> valores é a forma mais barata de fazer duas equipes escreverem testes incompatíveis. **A banda
> vigente é a de 4.13.6: `N_final ∈ [700, 900]`**, medido `774` **[M]** — a banda original também
> conteria o valor medido, de modo que nada de substantivo muda; o que muda é haver **um** enunciado
> em vez de dois.

**(C7) Detecção de runaway — O RUNAWAY OCORRE, E É O RESULTADO.** `t_runaway :=` primeiro `t` com
`max_i m_i / M_real >= 0.10`. Deve ser reportado e **marcado em toda figura**.

> **A regra que esta cláusula sempre teve — *"se o runaway ocorrer, ele é o resultado e deve ser
> relatado como tal; o que não é permitido é que ocorra sem ser detectado"* — está sendo HONRADA em
> 2026-08-08, e não contornada.** Ele ocorreu, foi detectado, e está medido:
> `max_i m_i / M_real = 0.3213`, `t_runaway ≈ 1.55 t_ff`. **[M]**
>
> A predição de campo médio que esta cláusula mandava confrontar (`~0.005`, `20x` abaixo do limiar)
> está **refutada por fator `64`**, e a Seção 4.12 registra por quê. **(C7) deixa de ser um critério
> a passar e passa a ser uma medição a publicar.**

**Obrigação que permanece, e é a parte que importa.** A partir de `t_runaway`, `V_CHAR`, `L_SCALE` e
a própria escolha de `eps` **deixam de descrever o sistema** — elas supõem que nenhum corpo domina, e
o raio de contato do corpo grande (`R = 0.034 m`) é `10%` de `r_half,min`. Toda figura da fase
colisional marca `t_runaway`; todo texto declara que além dele o objeto simulado não é mais o colapso
frio de N corpos parametrizado por essas escalas.

**(C8) Predição de `N_final`** — ver a Seção 4.13.6 para a predição vigente. Informativa, não
bloqueante.

---

## 7. Tolerâncias

Todas as tolerâncias abaixo são **relativas e adimensionais**, conforme a regra sem exceção da Seção
6.3 de `integradores.md`. Nenhuma foi ajustada a saída observada; cada uma traz a origem.

| identificador | grandeza | fp64 | fp32 | origem |
|---|---|---|---|---|
| `TOL-MASS-SUM` | `\|Δ sum_i m_i\| / M_real` por passe | `n_events * eps_prec` | idem | um arredondamento por evento (4.5, 4.9) |
| `TOL-EVENT-CONS` | `\|Δ(K+U+E_int)\| / \|E_0\|` através de um mapa de desfecho | `1e-5` (teto) e `1e-8` (piso), **só enquanto `max_i m_i / M_real < 0.10`** | `1e-4` | `E_int` é fechada ao nível do par e omite terceiros; resíduo medido `2.5e-6` **[M]**, margem `4x`. O piso impede o teste de virar vazio (`INV-23(a)`). **Escopo declarado em 2026-08-08 (d):** o resíduo escala com o termo mútuo (`∝ m_i m_j`), logo a normalização por `\|E_0\|` só é válida sem corpo dominante — ver `INV-23(a)` |
| `TOL-EVENT-INV` | conservações exatas por desfecho (`INV-20/21/22`) | `100 eps_prec` | `100 eps_prec` | reduções de `~10` termos; margem `~100x` sobre o medido (`5e-17`) |
| `TOL-EVENT-PRED` | destruições previstas (`ΔK = -T_cm`, `ΔL = -mu dr x u`) | `1e-12` | não testável | medido `1.01e-15`; margem `~1000x` |
| `TOL-EINT-DRIFT` | `\|ΔE_total/E_0\|` acumulada em `3 t_ff` com fusão | `<= 1e-2` **só enquanto `max_i m_i / M_real < 0.10`**; **REPORTADO** depois | idem | `~500` eventos x `2.5e-6` de resíduo de sinal único = `1.2e-3` **[M]**; margem `~8x`. **Escopo declarado em 2026-08-08 (d)** — ver a nota abaixo. Medido na execução aceita: `8.59` **[M]**, com `3280` eventos e um corpo de `321 m_bar` |
| `TOL-VIRIAL` | `\|2K/\|U\| / Q - 1\|` | `1e-12` | `1e-5` | exato por construção; cobre redução de `N` termos |
| ~~`TOL-COURANT`~~ | ~~`max C_coll <= 1`~~ | **REMOVIDA** | — | **[M]** `C_coll = 1.81` e `0.45` produzem a mesma física dentro de `0.5%` (4.4.3). `c_coll_max` é reportado, não limitado |
| ~~`TOL-REJECT`~~ | ~~`f_reject <= 0.05`~~ | **REBAIXADA** | — | linha de atenção editorial, não critério de teste (4.5). Medido `0.00407` **[M]** |
| ~~`TOL-EINT-NEG`~~ → ~~`TOL-EINT-MAG`~~ | `max_t \|E_int\| / \|E_0\|` | **REPORTADO** | idem | **2026-08-08: deixou de ser cota.** Três formulações, três reprovações; elevar a cota uma quarta vez seria o ajuste post-hoc que este documento proíbe. `\|E_int\| >> \|E_0\|` é a **assinatura do runaway aceito** (4.13.5), não um defeito. Medido `10.83` **[M]** |
| `TOL-PROB` | `\|sum_c p_c - 1\|` | `4 eps_prec` | `4 eps_prec` | três somas e uma divisão |
| `TOL-PROB-SYM` | `\|p_fus(x) - p_frag(1/x)\|` | `100 eps_prec` | `100 eps_prec` | **novo**: identidade algébrica exata do mapa `(1/x,3,x)/Z` (`INV-25`, cláusula 5) |
| `TOL-PROB-UNIF` | `\|p_c - 1/3\|` no controle uniforme | `4 eps_prec` | `4 eps_prec` | **apertado**: o controle é agora `(1/3,1/3,1/3)` literal, não um limite de `w`; não há mais `b/w` residual a cobrir |
| `TOL-PROB-MIN` | `min_c p_c` nos extremos além do clamp | `>= 1e-25` | `>= 1e-25` | **novo**: `X_CLAMP^-2 = 1e-24` **[T]**, acima do menor normal de fp32 (`1.18e-38`) |
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
  **parâmetro** (`chi`, `v_coh`), não a cota. `w`, `b` e `dt` saíram dessa lista: os dois primeiros
  não existem mais, e `dt` deixou de ser uma alavanca de validade colisional (Seção 4.4).
- **Os afrouxamentos desta revisão foram feitos ANTES de ver qualquer resultado do estágio 3**, e
  cada um deriva de uma medição do estágio 2 registrada neste documento. Isso é o oposto de afrouxar
  uma cota para fazer um teste passar, e a distinção deve permanecer visível: um afrouxamento
  posterior, motivado por uma falha observada, continua proibido.
- **O ESCOPO DE `TOL-EVENT-CONS` E `TOL-EINT-DRIFT`, declarado em 2026-08-08 (d) — e por que ele
  não é um afrouxamento.** Nenhum dos dois valores mudou. O que se declarou é a **região de
  validade da derivação**, que sempre foi: pares de massa comparável, resíduo de terceiros medido
  em `2.5e-6 |E_0|` por evento no pico de compressão do estágio 2. Esse resíduo é uma fração
  aproximadamente fixa (`1.2%` **[M]**) do **termo mútuo do par**, e o termo mútuo escala com
  `m_i m_j`. Com um corpo de `321 m_bar`, o termo mútuo de um evento chega a `~0.03 |E_0|` e o
  resíduo esperado a `~4e-4 |E_0|` **[T]** — `40x` o teto de `TOL-EVENT-CONS`. **A cota não foi
  violada por defeito de implementação; a premissa da sua normalização deixou de valer.**
  A alternativa honesta seria reduzir o número, e ela seria pior: uma cota fixa em `|E_0|` não é
  formulável num modelo cuja massa máxima não tem teto. Por isso o tratamento é o mesmo dado a
  `INV-23(c)`, `TOL-COURANT` e `TOL-REJECT` — **mudar o tipo do critério onde a sua derivação
  acabou, e não o seu valor para caber o número observado.**
- **O que fica em aberto, e é medição, não redação.** A forma escala-invariante da cota por evento —
  `|Δ(K+U+E_int)| / E_grav(m_i,m_j,d_ij) <= 5e-2` — é derivável do `1.2%` já medido, mas **não foi
  medida no regime pós-runaway** e está **[A]**. A medição que a converte em **[M]** é barata: um
  histograma dessa razão sobre os `3280` eventos da execução da Seção 4.13.4. **Até que ela exista,
  nenhum teste pode ser escrito contra `TOL-EVENT-CONS` numa execução que atravesse `t_runaway`.**

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
CHI_DEFAULT           = 0.1              # [M] FIXADO: N_coll/particula = 0.938 medido (Sec. 4.1.1)
R_REF_DEFAULT         = 5.0e-3           # m = CHI_DEFAULT * SOFTENING
N_STEPS_COLLISION     = 12600            # 3 t_ff a DT_COLLAPSE (massas iguais) -- era 50400
MAP_X_CLAMP           = 1.0e12           # [T] clamp de x; menor prob = X_CLAMP^-2 = 1e-24
MAP_ELASTIC_WEIGHT    = 3.0              # [T] a constante do numerador elastico em (1/x, C, x)
COLLISION_DRAWS_PER_EVENT = 2            # [T] normativo, Sec. 4.7.1 -- fixo para TODO canal
FRAG_F_MIN            = 0.1              # f = FRAG_F_MIN + (1 - 2*FRAG_F_MIN) * u2
FRAG_ETA              = 0.0              # fracao dissipada na fragmentacao (desligada)
FRAG_K_MAX            = 0.70             # E[max(f,1-f)] = 3/4 - FRAG_F_MIN/2, EXATO
# MASS_CAP_UNIFORM / MASS_CAP_DEFAULT: RETRATADAS em 2026-08-08.  A formula da Sec. 4.12 da
#   m* em unidades do PARCEIRO, nao de m_bar, e a fragmentacao 2->2 nao e' sumidouro de massa.
#   NAO HA TETO.  Valores antigos (5.667 e 3.17) preservados so' no texto, como registro do erro.
MASS_CAP_MEASURED     = 321.26           # [M] max_i m_i / m_bar de fato observado em 3 t_ff
COLLISION_SEED        = 20190225         # quarto fluxo, separado

# REMOVIDO em 2026-08-07 (b): v_esc_eff e todo o termo gravitacional de E_bind.
#   x = |u|^2 / v_coh^2 , v_coh = V_CHAR.  Ver Sec. 4.6.1: o termo CAUSAVA o runaway
#   (max m_i = 321 m_bar medido) em vez de conte-lo, e o piso x >= 1 que ele existia
#   para corrigir nunca existiu (o piso real era ~0.4).  PISO item 6 retirado.

# valores medidos no estagio 2 (Sec. 4.1.1 / 4.4), fp64, chi = 0.1, dt = 5.0e-4, 3 t_ff
COLL_N_PER_PARTICLE   = 0.938            # [M] N_coll_per_particle no rebote
COLL_ENCOUNTERS_3TFF  = 1902             # [M] encontros em 3 t_ff
COLL_C_COLL_MAX       = 1.8137           # [M] reportado, NAO limitado
COLL_F_REJECT_TOTAL   = 4.07e-3          # [M] reportado, NAO limitado
COLL_U_MAX            = 36.3             # [M] m/s -- era [A] 30.0
CORE_NUMBER_DENSITY   = 2888.3           # [M] m^-3 -- refuta o ~1.4e3 de integradores.md Sec. 4.3
EINT_THIRDBODY_RESID  = 2.5e-6           # [M] residuo por evento, em unidades de |E_0|
EINT_DRIFT_3TFF       = 1.2e-3           # [M] deriva acumulada esperada, em unidades de |E_0|

# REMOVIDOS nesta revisao (nao sao mais simbolos do projeto):
#   DT_COLLISION        -- ver Sec. 4.4; usa-se DT_COLLAPSE = 5.0e-4
#   COH_VELOCITY_FACTOR -- ver Sec. 4.6; v_coh E V_CHAR, sem fator
#   MAP_B, MAP_W, MAP_S_CLAMP -- ver Sec. 4.7; o mapa nao tem parametro de forma
#   ENS_DISPERSION_MAX  -- ver Sec. 4.13; (C4) retirado, K_SEEDS = 4 nao sustenta variancia
#   TOL_COURANT_MAX, TOL_REJECT_MAX -- ver Sec. 7; rebaixados a numeros reportados

# --- ensemble (Secao 4.13)
K_SEEDS               = 4                # era 32; ver Sec. 4.13 para o que se perde
ENS_N_FINAL_1_MAX     = 0                # (C1) nenhuma semente pode terminar com N_final = 1
ENS_MIN_EVENTS_MEDIAN = 50               # (C2)
ENS_T50_MIN_TFF       = 1.0              # (C3)
ENS_T50_FRACTION      = 1.00             # (C3) todas as 4 sementes
ENS_CHANNEL_MIN       = 0.05             # (C5)
ENS_MIN_MASS_FLOOR    = 1.0e-3           # (C6) em unidades de m_bar
ENS_MIN_MASS_SEEDS    = 3                # (C6) em pelo menos 3 das 4 sementes
ENS_RUNAWAY_THRESHOLD = 0.10             # (C7) max_i m_i / M_real
# predicao VIGENTE, Sec. 4.13.6 -- descreve o runaway em vez de nega-lo. As tres anteriores
# ((700,800) e (750,900) para N_final, (2,8) para max_m) tentavam prever um TETO DE MASSA que
# este modelo estruturalmente nao tem (Sec. 4.13.4): a fragmentacao e 2->2 e conserva m_i+m_j,
# logo nao ha sumidouro de massa.  Medido: max m_i = 321.26 m_bar.
ENS_N_FINAL_PREDICTED  = (700, 900)      # [A] Sec. 4.13.6
ENS_RUNAWAY_TFF        = (1.2, 2.0)      # [A] t_runaway / t_ff
ENS_MAX_MASS_FRACTION  = (0.15, 0.60)    # [A] max_i m_i / M_real em 3 t_ff
ENS_RUNAWAY_MIN_TFF    = 0.5             # [A] duracao minima da transicao
ENS_EINT_MAG_RANGE     = (3.0, 40.0)     # [A] max_t |E_int|/|E_0|

# --- ENSEMBLE K_SEEDS = 4 EXECUTADO em 2026-08-08 (Sec. 4.13.7).  [M]
# Protocolo: semente de POSICOES fixa em SEED; varia so' a de colisao, COLLISION_SEED + k.
#   semente    N_final   max m/M_real   t_runaway/t_ff   duracao/t_ff   |E_int|/|E_0|
#   20190225     774        0.3213         1.9517           0.7021         10.833
#   20190226     783        0.3032         1.9874           0.6188          9.050
#   20190227     915        0.0161         nunca              --            0.026   <- |p| = NaN
#   20190228     798        0.2926         1.9636           0.5355          7.834
# fusoes [226, 217, 85, 202] ; elasticas [923, 837, 359, 951] ; frag [2131, 2142, 1002, 2256]
# agregado: 27.1% el / 6.4% fus / 66.5% frag , 11331 eventos
#
# 20190227 NAO E' MEDICAO FISICA: terminou com |p| = nan e 1446 eventos (44% do tipico).
#   Reexecutada 3x pelo mesmo caminho de codigo: N_final = 785, max m = 307.6 m_bar, sem NaN.
#   Excluida por regra PRE-DECLARADA (Sec. 4.13.6: "massa ou momento inexatos -> defeito").
#   Causa NAO identificada; nao reproduzida.  Ver Sec. 4.13.7.
ENS_MAX_MASS_MEASURED  = (0.3213, 0.3032, 0.2926, 0.3076)  # [M] a 4a e' a rerodada de 20190227
ENS_MAX_MASS_CV        = 0.04            # [M] dispersao entre sementes de COLISAO -- notavelmente
                                         #     pequena; sugere que o valor final e' fixado pela
                                         #     realizacao de POSICOES, que ficou fixa.  Ver 4.13.7.
#
# NAO CONFIRMADOS -- divergem do que 4.13.4 registra para a MESMA execucao (Sec. 4.13.7):
#   t_runaway ~1.55  contra  1.9517 medido no ensemble  (26% de diferenca)
#   duracao 1.43 t_ff (48%)  contra  0.7021 t_ff (23%)  (fator 2)
#   A trajetoria e' a mesma (N_final, max m e E_int batem ate' o ultimo digito), logo a
#   divergencia esta' na DEFINICAO/MEDICAO.  "duracao da transicao" nunca foi definida
#   operacionalmente neste documento.  NAO CITAR 1.55 nem 48% ate' remedir.
#
# NAO REPORTADO pela campanha, e INV-31(C6) exige: min_i m_i / m_bar >= 1e-3 em >= 3 de 4.

# medido no estagio 3 defeituoso de 2026-08-07 (Sec. 4.13.2), preservado para regressao:
#   max m_i / m_bar = 321.4 , E_int/|E_0| = -109.8 , N_final = 744 , canais 34.5/12.9/52.7 %
#   Se qualquer numero proximo de 321 reaparecer, v_esc_eff voltou a algum ramo do codigo.

# --- tolerancias novas (Secao 7)
TOL_EVENT_CONS_FP64   = 1e-5             # era 1e-13; E_int fechada ao nivel do par (Sec. 4.10)
TOL_EVENT_CONS_FP32   = 1e-4
TOL_EVENT_CONS_FLOOR  = 1e-8             # piso: abaixo disso terceiros estao sendo computados
TOL_EINT_DRIFT        = 1e-2             # |Delta E_total/E_0| acumulada em 3 t_ff com fusao
TOL_EVENT_INV_ULP     = 100.0            # multiplica eps_prec
TOL_EVENT_PRED        = 1e-12
TOL_VIRIAL_FP64       = 1e-12
TOL_VIRIAL_FP32       = 1e-5
# TOL_EINT_NEG / TOL_EINT_MAG: REMOVIDAS em 2026-08-08.  |E_int| e' REPORTADO, nao limitado --
#   e' a assinatura do runaway aceito (Sec. 4.13.5), nao um defeito.  Medido 10.83 |E_0|.
TOL_PROB_SYM_ULP      = 100.0            # multiplica eps_prec (simetria x -> 1/x)
TOL_PROB_MIN          = 1e-25            # min_c p_c alem do clamp
TOL_PROB_UNIF         = 4                # multiplica eps_prec; era 1e-5
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

Listadas com justificativa.

> **Estado das emendas, 2026-08-08 (d).** O texto anterior dizia "**Nenhuma foi aplicada por este
> documento**", e isso deixou de ser verdade nesta data. Estado corrente:
>
> | subseção | alvo | estado |
> |---|---|---|
> | 9.1 / 9.1.1 / 9.1.2 | `docs/api-contract.md` | **APLICADA em 2026-08-08 (d)** |
> | 9.2 | `docs/integradores.md` | **APLICADA em 2026-08-08 (d)** |
> | 9.3 | `src/nbody/observables.py` | fechada desde antes; sem trabalho pendente |
> | 9.4 | `docs/glossario.md` | **PENDENTE** |
>
> A regra de fundo não mudou: este documento continua sendo onde a emenda é **decidida e
> justificada**, e os documentos-alvo são onde ela é **aplicada**. A lista permanece aqui depois de
> aplicada, como registro do porquê — apagá-la reescreveria a história.

### 9.1 `docs/api-contract.md`

| item | emenda | justificativa |
|---|---|---|
| `nbody.initial_conditions` | acrescentar `random_sphere(n, radius, seed, mass_spectrum=None, virial_ratio=0.0, f_cut=0.5, mass_seed=..., vel_seed=..., dtype, device) -> State` | é a IC das Seções 2 e 3; `cold_sphere` permanece intocada e continua sendo a referência bit a bit de `INV-17` |
| novo módulo | `nbody.populations`: amostragem de massas (Seção 2.7) e de velocidades (Seção 3.3), expondo `mass_min_from_mean`, `sample_masses`, `sample_velocities`, `solve_sigma`, e `lambda` como retorno auxiliar | separa a geração estocástica da IC geométrica; sem `lambda` exposto, `INV-16(c)` é intestável |
| novo módulo | `nbody.collisions` — **assinaturas fixadas na Seção 9.1.1 abaixo**, não em prosa | a colisão é um mapa separável do integrador (Seção 4.5) |
| `integrate()` | parâmetro adicional **somente por palavra-chave** `collision: Optional[CollisionModel] = None` | mantém a assinatura atual válida; `None` é o caminho existente, exigido bit a bit por `INV-30` |
| `integrate()` | parâmetro adicional **somente por palavra-chave** `collision_rng: np.random.Generator \| None = None` — **ver 9.1.2, que é a parte que importa** | `INV-32` só é satisfazível dentro de **uma** chamada se o gerador for recriado a cada chamada; um chamador que avança a simulação em pedaços precisa **possuir** o gerador |
| `integrate()` | levantar `ValueError` quando `collision is not None and softening == 0.0` | `INV-28`; sem isso o campo de aceleração vira `NaN` (Seção 5.2) |
| `integrate()` | levantar `ValueError` quando `collision is not None and integrator != "velocity_verlet"` | a colisão é definida **dentro do drift** de Verlet (Seção 4.5); não há definição de onde ela entraria em `euler`, `symplectic_euler` ou `rk4`. **Divergência registrada em 2026-08-08 (d): o código já faz isto e o documento não pedia** |
| `integrate()` | levantar `ValueError` quando `collision is None and collision_rng is not None` | um gerador entregue a um caminho que não o usa é erro do chamador, não configuração silenciosamente ignorada — regra de "nenhuma falha vira resultado degradado" do `api-contract.md` |
| `integrate()` | **tipo de retorno passa a ser `State \| tuple[State, CollisionRunStats]`** | com `collision=None` devolve `State`, como antes (`INV-30` bit a bit); com colisão ligada devolve também os acumuladores do passe longo. **Divergência registrada em 2026-08-08 (d)** — ver 9.1.2 |
| `nbody.observables` | acrescentar `n_live(state)`, `mass_spectrum_summary(state)`, `scales_from_state(state)` | `N_live` e as escalas realizadas são necessários a `INV-15`, `INV-24` e `INV-31` |
| convenção `m = 0` | documentar que slots com `m = 0` são inertes e que `State` os admite; `State.__post_init__` **não** deve rejeitá-los | é o contrato da Seção 5; hoje não está escrito em lugar nenhum |
| `Backend` | nenhuma mudança | o kernel de força é literalmente o mesmo; `m = 0` já é tratado corretamente (verificado, Seção 5.1) |
| contratos de erro | acrescentar que o desfecho de colisão consome **exatamente dois** sorteios por evento aceito, na ordem `(t*, i, j)`, para **todo** canal | reprodutibilidade (`INV-19(c)`, `INV-32`, Seção 4.7.1) |

#### 9.1.1 Assinaturas de `nbody.collisions` — normativas e literais

> **Esta subseção fecha a lacuna que gerou `tests/_stage2_binding.py`.** A versão anterior dava
> apenas nomes em prosa e uma lista de campos de `CollisionModel` (`r_ref, v_coh, b, w, frag_f_min,
> eta, seed`) que **não corresponde ao que existe** — omite `m_bar`, e quatro dos campos listados
> deixaram de existir com o mapa sem parâmetros da Seção 4.7. Sem lista de parâmetros nem tipo de
> retorno, a suíte de testes teve de descobrir a API por `inspect.signature` sobre objetos vivos, o
> que é exatamente o acoplamento que o processo existe para evitar.
>
> **A partir daqui as assinaturas são normativas.** Uma vez que `resolve` esteja implementado
> conforme a tabela, `tests/_stage2_binding.py` pode ser **deletado**, e com ele todos os
> `pytest.skip(ContractGap)` de `tests/test_collision_detection.py` e
> `tests/test_collision_pairing.py`.

**JÁ IMPLEMENTADO — transcrito de `src/nbody/collisions.py`, e agora vinculante:**

| símbolo | assinatura |
|---|---|
| `CollisionModel` | `@dataclass(frozen=True)` com `r_ref: float = R_REF_DEFAULT`, `m_bar: float = PARTICLE_MASS` |
| `CollisionCandidates` | `@dataclass(frozen=True)` com `i, j, t_star, rel_speed, contact_radius_sum: Tensor` (1-D, mesmo comprimento) e `n: int` como `@property` |
| `AcceptedPairs` | como `CollisionCandidates`, mais `f_reject: float` (**do passe**, Seção 4.5) |
| `contact_radii` | `(m: Tensor, model: CollisionModel) -> Tensor` |
| `detect` | `(state, dt: float, model: CollisionModel) -> CollisionCandidates` |
| `pair_disjoint` | `(candidates: CollisionCandidates) -> AcceptedPairs` |

**Armadilha normativa de `detect`.** O parâmetro é um `State`, mas `state.v` é interpretado como a
**velocidade de meio passo `v^(n+1/2)`**, não como `v^n`. `detect` não tem como verificar isso. **O
chamador é obrigado a construir `State(r=r^n, v=v^(n+1/2), m=m)`.** Passar `v^n` produz um detector
que roda, não falha, e detecta os pares errados. Isto tem de estar na docstring e no
`api-contract.md`.

**~~A IMPLEMENTAR~~ — JÁ IMPLEMENTADO desde 2026-08-08 (`src/nbody/collisions.py`); conferido
símbolo a símbolo em 2026-08-08 (d) e a tabela abaixo corresponde ao código. Continua normativa:**

| símbolo | assinatura |
|---|---|
| `CollisionModel` (campos adicionais) | `v_coh: float` — em m/s, **sem sentinela**; o chamador o computa. `seed: int = COLLISION_SEED` |
| `v_coh_from_state` | `(state, radius: float) -> float`, devolvendo `V_CHAR = sqrt(G M_real / radius)` |
| `CollisionOutcome` | `@dataclass(frozen=True)` com `state: State` (após o passe, **drift completo**), `n_elastic: int`, `n_merge: int`, `n_fragment: int`, `delta_e_int: float` (fp64), `delta_l_spin: Tensor` (shape `(3,)`, fp64), `f_reject: float`, `c_coll_max: float` |
| `resolve` | `(state, dt: float, model: CollisionModel, accepted: AcceptedPairs, generator, softening: float) -> CollisionOutcome` |
| `collision_pass` | `(state, dt: float, model: CollisionModel, generator, softening: float) -> CollisionOutcome` — compõe `detect` + `pair_disjoint` + `resolve` e é o que `integrate` chama |

**Três pontos normativos sobre `resolve`, cada um fechando uma ambiguidade:**

1. **`resolve` é PURA quanto aos acumuladores.** Ela devolve os deltas **deste passe**
   (`delta_e_int`, `delta_l_spin`) e **não** mantém `E_int` nem `L_spin` correndo internamente. Os
   acumuladores de longo prazo pertencem ao laço de integração, em fp64. Motivo: estado mutável
   escondido num módulo é precisamente o que faz duas equipes divergirem, e torna `INV-23(a)`
   (resíduo por evento) impossível de isolar.
2. **`resolve` completa o drift.** `CollisionOutcome.state` já tem os não participantes com
   `r += dt*v` e os participantes avançados `dt - t*` após o mapa (Seção 4.5, passos 3 e 4). O
   chamador não faz drift nenhum depois.
3. **`softening` é parâmetro explícito**, como em `accelerations(r, m, softening)`, e não campo do
   modelo. `resolve` precisa dele para `E_bind` (Seção 4.6) e para `E_grav` (Seção 4.10); `detect`
   não precisa e não o recebe.

#### 9.1.2 `collision_rng` — o parâmetro, e o motivo, que é a parte que não pode ser perdida

**Acrescentado em 2026-08-08 (d), registrando uma mudança de assinatura já feita no código.**

```python
def integrate(state, *, integrator, backend, dt, n_steps,
              softening=SOFTENING, callback=None, callback_every=None,
              collision: CollisionModel | None = None,
              collision_rng: np.random.Generator | None = None,
              ) -> State | tuple[State, CollisionRunStats]
```

**Contrato.** `collision_rng` é **somente por palavra-chave**.

- **Omitido (`None`)**: `integrate` constrói `np.random.default_rng(collision.seed)` internamente e
  **reproduz bit a bit o comportamento anterior** para qualquer chamador que faça uma única chamada
  por execução. Nenhum resultado publicado muda.
- **Fornecido**: é o objeto que **o chamador mantém vivo entre chamadas**, e `integrate` consome
  dele sem nunca o recriar nem o semear.

**O MOTIVO — e é ele, não o parâmetro, que impede alguém de "simplificar" isto depois.**

`INV-32` exige que o fluxo de colisão consuma **exatamente `2` sorteios por evento aceito**, de
forma contínua ao longo da execução, e `INV-19(c)` exige reprodutibilidade bit a bit da execução
inteira. A versão anterior de `integrate` construía o gerador **a cada chamada**, a partir de
`collision.seed`. Enquanto a execução era uma única chamada, isso era indistinguível de um fluxo
contínuo. **Passou a não ser**: o visualizador em tempo real avança a simulação **em pedaços**, uma
chamada de `integrate` por quadro. Nesse padrão, cada chamada reiniciava o fluxo do mesmo `seed`, de
modo que:

```
fluxo pretendido (uma chamada de M passos):   u_1 u_2 u_3 u_4 u_5 u_6 ...
fluxo realizado  (M chamadas de 1 passo):     u_1 u_2 | u_1 u_2 | u_1 u_2 | ...
```

**Consequências, todas físicas e nenhuma delas visível como erro:** o sorteio de canal deixa de ser
independente entre quadros e passa a ser **periódico com o período do quadro**; as frações de canal
medidas por `INV-26` deixam de ser amostras do mapa; e a mesma semente produz resultados diferentes
conforme o tamanho do pedaço, o que é exatamente a falha que `INV-19(c)` existe para impedir.
`INV-32` continuava passando — ele é enunciado **por passe** — e por isso a violação não aparecia em
nenhum teste. **É uma quebra de reprodutibilidade que só é detectável comparando uma execução
fatiada com a execução inteira**, e esse é o teste que a acompanha.

> **Aviso a quem for simplificar.** `collision_rng` parece um parâmetro redundante: `CollisionModel`
> já tem `seed`, e o valor padrão reproduz o comportamento antigo. **Removê-lo, ou fazer `integrate`
> re-semear a partir de `model.seed` quando ele é fornecido, reintroduz exatamente o defeito acima**,
> e reintroduz sem falhar teste nenhum da suíte atual. Se a remoção for proposta, o ônus é
> apresentar o teste de equivalência fatiada/inteira passando sem o parâmetro.

**Teste que fecha o buraco (a escrever; não existe hoje).** Comparar, bit a bit, o estado final e as
contagens de canal de: (i) uma chamada de `integrate` com `n_steps = M`; (ii) `M` chamadas de
`n_steps = 1` encadeadas, passando **o mesmo** `Generator`. Os dois têm de coincidir por `torch.equal`.
Repetir com o gerador **não** passado em (ii) e exigir que os dois **divirjam** — sem essa segunda
metade, o teste passa vazio contra uma implementação que ignore o parâmetro.

**Divergências entre este documento e `src/`, registradas e NÃO corrigidas por este documento**
(regra da Seção 1: divergência se relata, não se conserta de um lado só):

| # | o que o código faz | o que o documento dizia | leitura |
|---|---|---|---|
| 1 | `integrate` devolve `tuple[State, CollisionRunStats]` quando `collision is not None` | `-> State`, sem ressalva | **mudança de contrato não registrada.** Registrada agora na tabela de 9.1. Não é defeito físico; é o único jeito de o laço devolver os acumuladores que a Seção 9.1.1 (ponto 1) exige que **ele**, e não `collisions.py`, mantenha |
| 2 | `integrate` recusa `collision` com integrador diferente de `velocity_verlet` | não pedido em lugar nenhum | **o código é mais estrito que o documento, e está certo**: a Seção 4.5 define a colisão dentro do drift de Verlet e não define mais nada. Registrado agora como requisito |
| 3 | `integrate` recusa `collision_rng` sem `collision` | não pedido | correto pelo mesmo princípio; registrado |
| 4 | `collisions.regime_probabilities(x, *, elastic_weight, x_clamp)` é pública | 9.1.1 não a lista | é o mapa da Seção 4.7 exposto diretamente, e `INV-25` precisa dela para ser testável sem passar por `resolve`. **Deve entrar na tabela normativa de 9.1.1**; não entrou ainda porque a assinatura não foi fixada por este documento |
| 5 | `CollisionModel.v_coh` é `field(kw_only=True)` **sem valor padrão** | 9.1.1 diz "sem sentinela; o chamador o computa" | **concorda em substância** — obrigatório e sem padrão é a leitura correta de "sem sentinela". Registrado para que ninguém "conserte" pondo um padrão |

### 9.2 `docs/integradores.md`

| seção | emenda | justificativa |
|---|---|---|
| §2 (nova subseção 2.6) | relação entre softening e raio de contato: com `R_i+R_j < eps` as colisões ocorrem **dentro** da região regularizada, e a velocidade de impacto e o poço de par são os do potencial de Plummer, não os de massas pontuais; com `R_i+R_j > eps` o softening nunca atua e o sistema muda de regime (Seção 4.2 deste documento) | a semântica de `eps` muda qualitativamente quando há contato, e §2.2 não cobre esse caso |
| §7, `INV-10` | `U_MIN_BOUND` deixa de ser `-G m² N(N-1)/(2 eps)` e passa a `-G (M_real² - sum_i m_i²) / (2 eps)` | com massas desiguais a fórmula atual não é a cota; ela subestima ou superestima conforme a variância das massas |
| §7, `INV-9` | a definição operacional de `r_half` passa a ser a mediana de **massa** (Seção 5.3 deste documento); registrar que para `N = 1000` com massas iguais o valor é **bit a bit inalterado** e `COLLAPSE_R_HALF_MIN = 0.3472` continua válido. **A mudança de código já está feita**; falta só a emenda ao texto | a fórmula antiga media a mediana errada assim que as massas diferem |
| **§4.3, densidade de núcleo** | **corrigir `~1.4e3 m^-3` para `2888 m^-3`** e a separação local `~0.089 m` para `~0.0703 m` | **[M]**, medido no estágio 2 (Seção 4.1.1 deste documento): o valor de `integradores.md` subestima a densidade de núcleo por um **fator `2`**, e é ele, não o cálculo direto, que estava errado. Toda taxa de colisão deste documento depende disso |
| §7, `INV-9`, nota de amostragem | registrar que `COLLAPSE_R_HALF_MIN = 0.3472` é um mínimo **subamostrado** (a cada `OUT_DT = 1e-2 s`, isto é `20` passos); amostrado a cada passo o mínimo é `0.3457`, `0.4%` mais fundo **[M]** | evita que alguém leia a diferença como discordância entre execuções |
| §7, `INV-3` e `INV-4` | registrar que **não se aplicam** a execuções com fusão ou fragmentação, e nomear os substitutos (`INV-23`, `INV-24`) | sem essa ressalva, um teste correto de `integradores.md` falha contra uma implementação correta com colisões |
| §8.4 ("o que não pode ser afirmado") | acrescentar: nada sobre relaxação de dois corpos ou segregação de massa a partir de execuções com espectro de massas em `3 t_ff` — o tempo de relaxação por segregação é menor que o de relaxação geral por `~m_max/<m> = 285`, mas ainda **[A]** e não medido | com massas desiguais surge a tentação de afirmar segregação, que o horizonte não sustenta |
| §10 | acrescentar item: as extensões estocásticas estão em `docs/simulacao-estocastica.md`; `INV-1..INV-10` permanecem em vigor com as emendas acima | rastreabilidade |

### 9.3 `src/nbody/observables.py` — FECHADO

`half_mass_radius`. **O apontamento foi confirmado e a correção está IMPLEMENTADA E COMMITADA.**
`src/nbody/observables.py:70` usa `argsort` + `cumsum` + `searchsorted`, isto é, a mediana de
**massa** da Seção 5.3. Compatibilidade bit a bit para massas iguais e `N` par; nenhum valor
publicado mudou (`IC_R_HALF_0 = 4.881251`, `COLLAPSE_R_HALF_MIN = 0.3472`).

**Este item não tem trabalho pendente.** `INV-29` permanece como teste de regressão.

### 9.4 `docs/glossario.md`

Acrescentar entradas: **razão virial**, **maxwelliana truncada** (com a distinção `f(v)` × `p(|v|)`
da Seção 3.1, que é o ponto mais fácil de enunciar errado em todo o projeto), **raio de contato**,
**detecção varrida**, **energia interna acumulada**, **lei de potência truncada**. A entrada de
amolecimento de Plummer deve ganhar a ressalva da Seção 4.2 sobre colisões dentro da região
regularizada.

Acrescentar também, por causa desta revisão: **número de Courant colisional** (com a ressalva de que
é reportado e não limitado), **parâmetro de regime `x`**, e **passo fixo do fluxo de colisão**.

### 9.5 O reenquadramento de 2026-08-07 — por que este documento foi afrouxado

**Esta subseção é a memória institucional da revisão. Ela existe para que ninguém, daqui a um ano,
leia as simplificações abaixo como descuido.**

**O que mudou no critério.** O usuário reenquadrou o projeto explicitamente: **o produto é a
VISUALIZAÇÃO**, e o trade-off foi decidido por ele — **aceita-se maior erro em troca de menor
complexidade**. O papel da revisão física deixou de ser "exigir o modelo mais defensável" e passou a
ser "dizer qual é o piso: o modelo mais barato que ainda conserva o que não pode ser violado".
Consequência operacional: uma aproximação grosseira, uma constante escolhida a dedo ou um modelo
fenomenológico **deixaram de ser motivo de veto**; violar uma conservação exata continua sendo.

**Onde o piso está: Seção 4.14.** Eram nove pontos; a revisão (b) retirou o item 6 e são **oito**
(Seção 9.6). Fora deles, há liberdade de projeto.

**Registro das mudanças, com o que cada uma custou:**

| # | mudança | o que se ganhou | o que se perdeu |
|---|---|---|---|
| 1 | `DT_COLLISION` removido; `dt = 5.0e-4` | `4.2x` no tempo de toda campanha e do visualizador em modo colisional | `0.47%` na contagem de colisões **[M]**; `C_coll,max` sobe de `0.45` a `1.81` |
| 2 | `C_coll` e `f_reject` rebaixados a números reportados | dois critérios de invalidação que não discriminavam nada saem do caminho | nenhuma detecção automática de `dt` inadequado — mas ela nunca funcionou, ver 4.4 |
| 3 | `chi = 0.1` fixado por medição mínima | campanha completa (`5` execuções, grade de `chi` x população) cortada para **uma** de `78 s` | a população com espectro de massas não foi medida; a figura de distribuição temporal não foi produzida |
| 4 | mapa `(1/x, 3, x)/Z` | três constantes de forma (`MAP_B`, `MAP_W`, `MAP_S_CLAMP`) e a exigência de *log-sum-exp* eliminadas; **zero parâmetros a calibrar** | `p_fus` no núcleo cai de `0.242` a `0.114`, metade das fusões; `INV-31(C5)` fica mais próximo do piso de `5%` |
| 5 | fragmentação ao longo da normal de contato | `3` sorteios por evento eliminados; consumo do fluxo fixado em `2` | a cláusula de isotropia de `INV-22` perdeu objeto e saiu |
| 6 | `E_int` fechada ao nível do par | de `O(N)` para `O(1)` por evento; `resolve` não precisa mais de acesso ao estado global para contabilizar | **`E_total` deixa de ser exatamente conservada**: deriva `~0.12%` de `\|E_0\|` por execução **[M]**; `TOL-EVENT-CONS` afrouxa de `1e-13` para `1e-5` |
| 7 | ensemble de `32` para `4` sementes | `~20 min` de campanha para `~2.5 min` | **(C1) e (C4) perdem poder estatístico**; (C4) foi retirado — bimodalidade agora só é vista por inspeção |
| 8 | `COH_VELOCITY_FACTOR` removido | um parâmetro a menos; `v_coh` **é** `V_CHAR` | nenhuma — o valor justificado sempre foi `1.0` |

**O que NÃO foi afrouxado, e por quê.** A Seção 4.10 (termo mútuo em `E_int`) manteve o seu veto:
sem ele a curva de energia do HUD vai visivelmente embora. **A Seção 4.6 também manteve o seu, e a
revisão (b) o derrubou** — ver abaixo.

### 9.6 A revisão (b) de 2026-08-07 — o que a execução do estágio 3 obrigou a mudar

Esta é a segunda revisão do mesmo dia, e é de natureza diferente da primeira: a revisão (a) foi uma
**simplificação deliberada**, decidida antes de qualquer resultado. A revisão (b) é uma **correção de
erro**, forçada por uma predição que falhou (Seção 4.13.2).

| # | mudança | por quê | o que se perdeu |
|---|---|---|---|
| 1 | **Termo gravitacional retirado de `E_bind`**; `x = \|u\|²/v_coh²` | `v_esc_eff² ∝ M` fazia `x -> 0` e `p_fus -> 1` conforme o corpo crescia: realimentação positiva. Medido `max m_i = 321 m_bar`, `90x` a predição | nada; a correção **remove** um termo, uma raiz e uma soma por evento |
| 2 | **PISO item 6 retirado** | a sua justificativa (`x >= 1` sem o termo) era falsa: o piso real era `~0.4` | o piso passa de nove a oito itens |
| 3 | **Sinal da fragmentação corrigido em 4.10** | erro **deste documento**, fielmente implementado; causa direta de `E_int = -109.8` | nada; era um erro |
| 4 | **`INV-23(c)` passa de cota de sinal a cota de magnitude** (`\|E_int\| <= 1 \|E_0\|`) | a fusão injeta legitimamente energia neste modelo (4.10); uma cota de sinal reprova código correto | perde-se a detecção automática de injeção — que nunca foi um sintoma válido |
| 5 | "fusão sempre dissipa" e "corpos massudos se auto-regulam" **retratadas** | ambas descendiam da mesma premissa falsa | duas afirmações a menos, ambas erradas |

**A raiz única.** Os cinco itens acima, mais o erro de análise da Seção 4.12, descendem todos de
**uma única premissa falsa**: a de que o par chega ao contato com a energia de uma **queda desde o
infinito**. Num núcleo suavizado a queda é desde a separação interpartícula local, que já está dentro
do raio de suavização, e entrega apenas `~40%` daquela energia (Seção 4.6.1). A premissa apareceu
**três vezes** no documento — em 4.6 (argumento 1), em 4.10 ("fusão sempre dissipa") e em 4.8
(decomposição em `|u_inf|`) — e as três sobreviveram à revisão (a) porque nenhuma foi confrontada com
a dinâmica medida. **Lição transferível: uma premissa reutilizada em três lugares é uma premissa que
precisa ser medida uma vez.**

**Regra que continua valendo, sem exceção — e como a revisão (b) se enquadra nela.** Os
afrouxamentos da revisão (a) foram decididos **antes** de qualquer resultado do estágio 3, e cada um
deriva de uma medição do estágio 2 registrada aqui. A revisão (b) veio **depois** de um resultado, e
por isso teve de passar por um teste mais duro, aplicado explicitamente na Seção 4.6.2: a alavanca
trocada tem de atacar a **causa identificada**, e teria de ter sido a escolha certa caso o mecanismo
fosse conhecido antes. Ambas as condições foram verificadas e a verificação está escrita.
**Afrouxar uma cota para fazer um teste passar continua proibido**; o que a revisão (b) fez foi
retirar um termo errado e apertar a análise, não relaxar um critério para acomodar um número.

### 9.7 A revisão (c) de 2026-08-08 — quando parar de mexer no modelo

A revisão (c) **não muda o modelo**. Ela muda a predição e o estatuto de dois critérios, e o seu
conteúdo principal é uma decisão de parada.

**O que a segunda execução mostrou.** As correções da revisão (b) funcionaram (`E_int` melhorou
`10x`), **e o runaway ocorreu assim mesmo, com valor final praticamente idêntico** (`321.26` contra
`321.4`). Um segundo mecanismo, portanto, e não na fórmula do mapa.

**O mecanismo, decidido por contagem e não por hipótese.** Houve `226` fusões na execução inteira
(`6.9%` de `3280`, confirmado pela queda `N: 1000 -> 774`). Montar `321 m_bar` a partir de corpos de
`1 m_bar` exige `320` fusões numa árvore binária. `320 > 226`: **a fusão não podia ter construído o
corpo, logo a fragmentação aumentou a massa de algum corpo.** **[T]** A fragmentação é `2 -> 2` e
conserva `m_i + m_j` exatamente — **não retira massa do conjunto, só redistribui** — e **eleva** o
máximo sempre que o parceiro passa de `42.9%` do corpo grande em esperança. **Este modelo não tem
sumidouro de massa e, portanto, não tem teto de massa, para nenhum valor de nenhum parâmetro.**

**Por que a decisão foi aceitar, e não uma terceira correção.**

| critério | leitura |
|---|---|
| existe correção simples? | **não** — a causa é estrutural; conter exigiria fragmentação `2 -> muitos` ou supressão por massa, **ambas acrescentam complexidade** |
| o usuário aceita? | **sim** — "terminar em um corpo só não é falha, desde que não seja sempre nem rápido demais": `774` corpos sobrevivem, e a transição toma `48%` da execução |
| o que se vê na tela? | **melhor do que a alternativa** — na rodada 1 a transição tomava `1.2%` do tempo e lia como *glitch*; agora toma metade da execução e lê como **processo**: um objeto crescendo e comendo o núcleo, que é um *runaway merger* reconhecível |
| o alvo era do modelo ou da análise? | **da análise** — `max m ~ 3 m_bar` nunca foi propriedade deste modelo; era propriedade de uma aproximação de campo médio que falhou **três vezes** |
| o documento já tinha regra? | **sim, e ela mandava aceitar** — `INV-31(C7)`: *"se o runaway ocorrer, ele é o resultado e deve ser relatado como tal"* |

**As três falhas da Seção 4.12, na mesma família.** As três revisões encontraram o mesmo erro de
método em lugares diferentes: **um ponto fixo calculado com um coeficiente co-evolutivo mantido
fixo.** Na (b) o coeficiente congelado era `x`; na (c) é `m'`, a massa do parceiro. E por baixo dos
dois havia uma terceira coisa, estrutural, que nenhuma correção de coeficiente alcançava: a ausência
de sumidouro. **Lição transferível: antes de anunciar um teto a partir de um ponto fixo, verificar
(i) que os coeficientes não dependem da variável de estado e (ii) que existe um termo de perda.**

**Onde a regra anti-post-hoc mordeu, e foi respeitada.** `INV-23(c)` já tinha sido reformulado duas
vezes e reprovado três. **Elevar a cota uma quarta vez seria exatamente o ajuste proibido.** Em vez
disso ela foi **retirada da condição de cota**: `|E_int|` mede fielmente um fenômeno que o projeto
decidiu aceitar, e um número que mede bem um resultado aceito não pode reprovar esse resultado.
A distinção — mudar o **tipo** do critério por argumento, em vez de mudar o seu **valor** para caber
o número — é o que separa esta decisão de um ajuste de resultado.

**O que ficava em aberto — e foi MEDIDO em 2026-08-08.** Toda a Seção 4.13.5 decidiu sobre **uma**
execução. `INV-31` com `K_SEEDS = 4` foi executado e **a decisão está apoiada**: três sementes de
colisão limpas (quatro, com a rerodada da que falhou) produzem runaway, com
`max m / M_real = 0.306 ± 0.012` — dispersão de `4%`. Não há bimodalidade entre sementes de colisão.
Resultados, vereditos e ressalvas na **Seção 4.13.7**.

**O que continua em aberto, e é a próxima medição.** O ensemble variou **apenas a semente de
colisão**, com a realização de posições fixa. A robustez **entre realizações da condição inicial**
não foi testada — e a dispersão pequena entre sementes de colisão é indício de que é justamente a
realização de posições que fixa o valor final, o que torna essa medição **mais** importante, não
menos. Além dela, ficam pendentes `INV-31(C6)` (não reportado) e a remedição de `t_runaway` e da
duração da transição, cujos valores publicados divergem do ensemble para a mesma execução.

### 9.8 A revisão (d) de 2026-08-08 — reconciliação, não modelagem

**Esta revisão não muda o modelo, não muda nenhum valor de tolerância e não introduz nenhuma
decisão física nova.** Ela faz três coisas: propaga decisões já tomadas para as seções que ainda
exibiam o texto anterior, declara o regime de validade de duas cotas cuja derivação a execução
aceita deixou para trás, e registra as emendas de assinatura que o estágio 5 obrigou.

**Por que ela foi necessária, e é a lição transferível.** As revisões (a), (b) e (c) mexeram cada
uma em pontos diferentes do documento e **corrigiram o lugar onde a decisão foi tomada, não todos os
lugares onde ela era repetida**. O resultado é um documento que se contradiz: a Seção 4.10 retirou
`INV-23(c)` da condição de cota e a Seção 6 continuou exibindo `min_t E_int >= -1e-2 |E_0|` com um
bloco "Se (c) falhar. Não afrouxar a cota". **Quem escreve teste lê a Seção 6.** Um invariante que o
próprio documento revoga noutra seção é pior que invariante nenhum: ele produz um teste vermelho
contra código correto, e o reflexo seguinte é mexer no código. **Num documento normativo, retratar
uma decisão é trabalho de varredura, não de parágrafo.**

**O que foi reconciliado:**

| # | onde | o que sobrevivia | resolvido para |
|---|---|---|---|
| 1 | Sec. 6, `INV-23(c)` | cota `min_t E_int >= -1e-2 \|E_0\|` e o bloco "Se (c) falhar" | reportado, não bloqueante; `[M] 10.83` |
| 2 | Sec. 6, `INV-23(b)` | critérios de `INV-4` sobre `E_total`, sem regime | válido só antes de `t_runaway` |
| 3 | Sec. 6, `INV-23(a)` / Sec. 7 | cotas normalizadas por `\|E_0\|`, derivadas de pares de massa igual | escopo declarado; forma escala-invariante marcada **[A]** |
| 4 | Sec. 4.10 | "é por isso que `INV-23(c)` existe" | a injeção legítima é razão para **não haver** cota |
| 5 | Sec. 4.10 | "contido o runaway, o maior corpo fica em `~3 m_bar` e a patologia desaparece sozinha" | retratado; e o caso "patológico" **é** o mecanismo do runaway |
| 6 | Sec. 4.10 | bloco da revisão (b) sem marcador, lendo como norma | marcado `[HISTÓRICO]` |
| 7 | Sec. 4.11 | degrau de evento "`~1e-3 \|E_0\|`" | escala com `m_i m_j`; sem cota superior |
| 8 | Sec. 4.11 `(D1)` | "conservado exatamente pelos eventos, mede só o integrador" | falso nas duas metades; regime declarado |
| 9 | Sec. 1 | "as **nove** coisas do PISO" | oito desde a revisão (b) |
| 10 | Sec. 1 | "este documento **é escrito** antes de a implementação existir" | nota de estado |
| 11 | Sec. 1.3 | estágio 3 "próximo" | executado duas vezes; **ensemble executado, ver 4.13.7** |
| 12 | Sec. 4.12 | cálculo de teto retratado, redigido no presente | marcado `[HISTÓRICO]`; incoerência `3.53` × `3.17` registrada, não reconciliada |
| 13 | `INV-31` | **duas** cláusulas `(C8)`, com bandas diferentes | uma, a de 4.13.6 |
| 14 | Sec. 10, item 19 | "não há crescimento descontrolado, e há teto fechado" | retratado |
| 15 | Sec. 10, item 37 | "corrigido `x`, o ponto fixo é genuíno e vale `~3.2 m_bar`" | retratado |
| 16 | Sec. 10, item 39 | cota de magnitude `\|E_int\| <= 1` | superseder apontado (item 45) |
| 17 | Sec. 9 | "nenhuma emenda foi aplicada por este documento" | tabela de estado |

**O padrão, dito uma vez para não precisar ser redescoberto.** Todas as dezessete sobrevivências
são do mesmo tipo: **o documento retratou a conclusão e manteve a frase que a usava como premissa.**
Nenhuma delas foi encontrada por leitura sequencial — foram encontradas procurando, para cada
decisão retratada, **todos** os lugares onde o seu número ou o seu nome aparecia. Esse é o
procedimento, e ele é obrigatório em qualquer revisão futura que retrate algo.

**Um erro DESTA revisão, corrigido no mesmo dia e registrado em vez de apagado.** Uma versão
anterior da Seção 1.3 afirmava que o ensemble `K_SEEDS = 4` **"não foi executado"**. Ele foi
executado, e os resultados estão em 4.13.7. **Eu inferi o estado da campanha a partir do estado do
documento, em vez de verificar** — que é, literalmente, a forma de erro que esta revisão passou
dezessete itens a catalogar: uma premissa não confrontada com a medição. A distinção que a
afirmação errada tentava fazer era **válida** e sobreviveu à correção, em forma correta: variar a
semente de colisão com a condição inicial fixa **não** é variar a realização. Ela está em 4.13.7,
"O que o ensemble testa, e o que ele NÃO testa". **Ficar certo pela razão errada continua sendo
ficar errado**, e é por isso que o episódio está escrito aqui.

**O que esta revisão deliberadamente NÃO fez.** Não reconciliou os números internos dos cálculos
retratados (item 12): dar coerência a um cálculo que não é predição é conferir-lhe um estatuto que
ele não tem. Não inventou uma cota nova para substituir `TOL-EVENT-CONS` no regime pós-runaway: a
forma correta está enunciada e marcada **[A]**, com a medição que a decide nomeada. E não tocou em
`src/` nem em `tests/`: as cinco divergências entre documento e código estão **registradas** em
9.1.2, não corrigidas.

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
11. **A álgebra da detecção varrida está correta**, verificada contra minimização por grade.
    `C_coll` **excede `1`** no regime perturbativo (`1.81` medido a `chi = 0.1`, `dt = 5e-4`), mas
    isso **não** causa perda de colisão — ver o item 26. **Este item foi revisado: a conclusão
    anterior, `DT_COLLISION = 1.25e-4`, está revogada.** `dt = DT_COLLAPSE = 5.0e-4`, e `C_coll` é
    um número reportado, não um invariante.
12. **A colisão entra dentro do *drift*, não depois do passo.** É a única forma de aplicar o impulso
    na configuração de contato, e é a paralelidade entre impulso e separação **no ponto de
    aplicação** que faz a colisão elástica conservar `L` exatamente. **"Separar até contato exato"
    está vetado**: injeta energia; no esquema adotado a sobreposição nunca se forma.
13. **O pareamento guloso conserva massa** e é uma **aproximação declarada** da sequência exata de
    eventos. O adiamento por disjunção é o **único** canal de perda de colisão do modelo (não há
    tunelamento, item 26), e é medido: `f_reject_total = 0.407%` **[M]**, reportado e não limitado.
    Desempate por `(i, j)` é normativo, sem ele não há reprodutibilidade entre dispositivos.
14. **A afirmação sobre a fusão está meio certa.** `ΔK = -(1/2) mu |u|²` **exatamente** —
    confirmado a `1e-15`. Mas `ΔU` **não** é só o termo mútuo: há termos de terceiro corpo que não
    se anulam e **não têm forma fechada**. **Revisado no item 32: em vez de computar esses termos em
    `O(N)`, esta revisão os OMITE e declara o resíduo medido** (`2.5e-6 |E_0|` por evento).
15. **A fusão e a fragmentação destroem momento angular** por `-mu (dr x u)` exatamente — o spin do
    par, que o modelo não representa. O enunciado do projeto não mencionava isso. Acumulador
    `L_spin` obrigatório, com `L_total = L_orb + L_spin`; sem ele o efeito atinge `~1e-4` de
    `L_SCALE` e nenhum invariante o veria.
16. **A construção de fragmentação está correta**: massa, momento e `T_cm` conservados exatamente,
    verificados — e continuam exatos com a saída ao longo da normal de contato (item 31), porque
    nenhuma dessas provas usa a direção de `u'`.
    Ressalva obrigatória: conservar `T_cm` exatamente significa **não dissipar nada** —
    isto não é fragmentação no sentido astrofísico, e é o que permite `E_int` decrescer.
17. **`E_int` é avaliada ATRAVÉS DO MAPA DE DESFECHO, com posições congeladas — nunca ao longo do
    passo.** Avaliá-la sobre o passo absorveria o erro de truncamento do integrador e tornaria
    `E_total` conservada trivialmente, destruindo o diagnóstico que ela existe para salvar. Emenda
    obrigatória à proposta original.
18. **A banda limitada de `|ΔE/E₀|` não sobrevive a fusão ou fragmentação.** Sobrevive: simpletismo
    entre eventos, `P`, e — só para colisões elásticas — `E` e `L` exatos. O substituto é um
    conjunto de quatro peças: `E_total = K+U+E_int` (agora com a deriva residual declarada de
    `~0.12%`, item 32), a curva `E_int(t)`, o resíduo por evento, e `L_total`. **A comparação entre
    integradores continua sendo feita sem colisões.**
19. **~~Não há crescimento descontrolado com os parâmetros padrão, e há teto fechado.~~
    RETRATADO EM 2026-08-08 (d) — ver os itens 41 a 44, e as Seções 4.12 e 4.13.4.** Esta era a
    sobrevivência mais consequente do resumo: um item redigido no presente do indicativo, no
    capítulo que existe para ser lido de relance, afirmando exatamente o contrário do que duas
    execuções mediram. **O enunciado vigente é: este modelo não tem sumidouro de massa e portanto
    não tem teto de massa, para nenhum valor de nenhum parâmetro** — a fragmentação é `2 -> 2`,
    conserva `m_i + m_j` e **concentra** massa em vez de contê-la. Medido `max_i m_i = 321.26
    m_bar`, `max_i m_i / M_real = 0.3213` **[M]**. A fórmula
    `m*/m_bar = (p_fus + k p_frag)/((1-k) p_frag)`, com `k = 3/4 - a/2 = 0.70`, permanece correta
    como ponto fixo a `p` constante e **em unidades do parceiro `m'`, não de `m_bar`** (item 44);
    os valores `~3.5` e `5.67` são o cálculo de campo médio que foi feito e falhou, e não são
    predição.
20. **~~O parâmetro de regime `x` exige o termo gravitacional.~~ RETRATADO em 2026-08-07 (b) — ver
    o item 35.** O veto estava errado nos seus três argumentos, e o terceiro estava invertido.
    `x = |u|²/v_coh²`, sem termo gravitacional. A faixa visitada é `x ∈ [~0.02, ~122]`, quase quatro
    décadas cavalgando `x = 1`, e **`x` não depende de massa alguma**.
21. **O mapa é uma interpolação fenomenológica, não uma teoria de colisões.** Seu conteúdo físico é
    a ordenação e a monotonicidade; a constante `3` e a escala `v_coh` são convenção. É permitido
    escrever "sob o modelo fixado, `X%` dos eventos foram fusões"; é **proibido** escrever "o colapso
    produz `X%` de fusões".
22. **`m = 0` é exatamente inerte em todos os caminhos verificados** — força, `U`, `K`, `P`, `L`,
    centro de massa —, bit a bit. **A única exceção é `eps = 0` com slot morto coincidente, que
    produz `NaN` via `0 * inf`**; daí a proibição normativa de colisões com `eps = 0`. Slots mortos
    ficam em `r = r_fundido`, `v = v_fundido`.
23. **`half_mass_radius` FOI corrigido** para a mediana de massa, e a correção está commitada
    (`src/nbody/observables.py:70`). Bit a bit compatível para massas iguais e `N` par, logo
    `IC_R_HALF_0 = 4.881251` e `COLLAPSE_R_HALF_MIN = 0.3472` permanecem válidos. **Item fechado.**
24. **O invariante de ensemble não rejeita o runaway; rejeita a degenerescência.** **`K = 4`**
    sementes (reduzido de `32`), com critérios sobre `N_final = 1`, número mínimo de eventos,
    `t_50 > 1 t_ff`, e cobertura dos três canais. **O índice de dispersão (C4) foi retirado**:
    variância sobre `4` amostras não é medição, e a bimodalidade passa a ser vista por inspeção
    direta. Isto é perda de poder de detecção, e está declarada.
25. **Todo `[A]` nomeia a medição que o converte em `[M]`, e três foram convertidos no estágio 2:**
    `chi = 0.1` (`N_coll_per_particle = 0.938`), a densidade de núcleo (`2888.3 m^-3`, que **refuta**
    o `~1.4e3` de `integradores.md` §4.3), e `|u|_max` (`36.3 m/s`). Nenhum `[A]` restante pode virar
    afirmação na prosa do relatório sem passar pela medição que ele mesmo declara.

### Decisões que esta revisão acrescenta

26. **Não há tunelamento no detector varrido, para nenhum `C_coll`.** Sobre uma reta, `dr . dv` é
    monotonicamente não decrescente, logo o passo que contém a aproximação máxima sempre tem
    `dr . dv < 0` no início e o `clamp` devolve o mínimo interior exato. **[T]**, verificado em
    `4.0e6` encontros com fração perdida `0.000000` **[M]**. **A colisão frontal é o caso mais fácil
    para o varrido, não o mais difícil.** `DT_COLLISION` foi derivado de uma premissa — resolução no
    fim do passo — que a Seção 4.5 proíbe, e está **removido**: `dt = DT_COLLAPSE = 5.0e-4`.
27. **`C_coll` e `f_reject` são números reportados, não invariantes.** `C_coll = 1.81` e `0.45`
    produzem a mesma física dentro de `0.5%` **[M]**; um número com essa propriedade não é condição
    de validade. `f_reject` passa a ser definido **por passe**, com `max` e `total` reportados — o
    que o torna testável de forma determinística, o que a definição antiga não era.
28. **O mapa de regime não tem parâmetro livre:** `(p_fus, p_el, p_frag) = (1/x, 3, x)/Z`. Sem
    `exp`, sem `log`, sem *log-sum-exp*. Conserva soma `1`, positividade estrita, monotonicidade,
    máximo elástico em `x = 1`, e ganha uma **simetria exata sob `x -> 1/x`** que o softmax não
    tinha. `MAP_B`, `MAP_W` e `MAP_S_CLAMP` removidos.
29. **O consumo do fluxo aleatório é fixo em `2` sorteios por evento aceito, para todo canal**, com
    `u2` sorteado **incondicionalmente** antes do desvio de canal. Sem isso, `INV-19(c)` é
    inatingível. Testável por equivalência de estado do `Generator` entre passes de canais diferentes
    (`INV-32`).
30. **O índice menor sempre fica com o corpo.** Fusão: `i` recebe o corpo fundido, `j` recebe
    `m = 0`, `r_j = r_i`, `v_j = v_i`. Fragmentação: fragmento `a` (massa `f M`) em `i`, `b` em `j`.
    A versão anterior não dizia qual slot sobrevive, o que tornava o determinismo indefinível.
31. **A fragmentação sai ao longo da normal de contato, não numa direção isotrópica sorteada.**
    Nenhuma prova de conservação usa a direção de `u'`, logo massa, `P`, `T_cm`, `K` e `sum m r`
    continuam exatos. A cláusula de isotropia de `INV-22` foi removida por perda de objeto;
    `INV-22` permanece em vigor.
32. **`E_int` é fechada ao nível do par, `O(1)` por evento**, e o termo de terceiro corpo é um
    resíduo **declarado e medido**: `2.5e-6 |E_0|` por evento, `~0.12%` acumulado em `3 t_ff`
    **[M]**. **`E_total` deixa portanto de ser exatamente conservada, e isso é o comportamento
    previsto de uma implementação correta** — um teste que exija conservação exata reprova código
    certo. O termo **mútuo**, esse, não pode sair: omiti-lo fabrica `~10%` de `|E_0|`.
    **Escopo acrescentado em 2026-08-08 (d):** os dois números acima são do regime de massas
    comparáveis do estágio 2. O resíduo é `~1.2%` do **termo mútuo do par** e portanto escala com
    `m_i m_j`; na execução aceita, com um corpo de `321 m_bar`, mediu-se `|ΔE_total/E_0| = 8.59`
    **[M]**. Ver a nota de escopo da Seção 7: **as cotas não mudaram de valor, mudaram de regime
    declarado**, e a forma escala-invariante da cota por evento está **[A]**, pendente de medição.
33. **O piso do modelo são oito pontos, e está na Seção 4.14.** Fora deles há liberdade de projeto.
    Dentro deles não há. Cada um traz o que quebra se for simplificado.
34. **Este documento foi deliberadamente afrouxado em 2026-08-07, e a Seção 9.5 registra por quê,
    o quê, e o que cada simplificação custou.** O produto é a visualização; aceita-se maior erro por
    menor complexidade. Todos os afrouxamentos foram decididos **antes** de qualquer resultado do
    estágio 3 e derivam de medições do estágio 2 registradas aqui. Afrouxar uma cota **depois** de
    ver uma falha continua proibido.

### Decisões da revisão (b), forçadas pelo estágio 3

35. **O termo gravitacional de `E_bind` CAUSAVA o crescimento descontrolado que se acreditava que
    ele contivesse, e foi retirado.** `v_esc_eff² ≈ 2GM/eps` cresce com a massa enquanto `|u|` fica
    na escala do núcleo, logo `x -> 0` e `p_fus -> 1`: realimentação positiva. `p_fus` cruza `1/2`
    em `m ≈ 22 m_bar` **[T]**, e a execução mediu `max m_i = 321 m_bar` **[M]**, `90x` a predição.
    **`x = |u|²/v_coh²`**, sem termo gravitacional e sem dependência de massa. O **item 6 do PISO
    está retirado** — era o único item do piso justificado por um comportamento previsto em vez de
    por uma conservação exata, e foi o único que caiu.
36. **A premissa de "queda desde o infinito" era falsa, e aparecia em três lugares.** No núcleo
    suavizado o par cai desde a separação interpartícula local (`0.070 m`, comparável a
    `eps = 0.05 m`) e chega com só `~40%` da energia de queda desde o infinito **[T]**. Disso
    decorrem, todas retratadas: o piso `x >= 1` (era `~0.4`), "fusão sempre dissipa" (ela pode
    injetar, `-0.034 |E_0|` num par `300:1`), e a decomposição de 4.8 em `|u_inf|`.
37. **A Seção 4.12 errou por congelar `p_fus` e `p_frag` num `x` fixo**, quando `x` dependia da
    própria massa em crescimento. A fórmula do teto está certa como ponto fixo a `p` constante; o
    que faltou foi verificar que `m -> m*(m)` tem cruzamento estável. Não tinha: o teto fugia à
    frente da massa. ~~Corrigido `x`, o ponto fixo é genuíno e vale `~3.2 m_bar`.~~ **Esta última
    frase está RETRATADA em 2026-08-08 (d)**: corrigir `x` removeu **um** dos dois defeitos, e o
    item 44 mostra que o restante (`m*` em unidades do parceiro) e a ausência de sumidouro de massa
    bastam para que não haja ponto fixo nenhum. Medido `321.26 m_bar` **com `x` já independente de
    massa** **[M]**. **Um ponto fixo calculado com coeficientes congelados só é um teto se os
    coeficientes não dependerem da variável de estado — e, mesmo então, só se existir um termo de
    perda** (Seção 9.7).
38. **A expressão de fragmentação de `E_int` estava com o sinal invertido neste documento**, e foi
    fielmente implementada: é a causa direta de `E_int/|E_0| = -109.8`. Corrigida para
    `E_grav(depois) - E_grav(antes)`. A magnitude, essa, era consequência do runaway — fragmentar
    `300 m_bar + 1 m_bar` multiplica `m_a m_b` por `75.5x`. **Duas causas, uma raiz; nenhum terceiro
    problema escondido.**
39. **[SUPERSEDIDO PELO ITEM 45 — não há mais cota alguma.]** `INV-23(c)` passou de cota de SINAL
    para cota de MAGNITUDE, `|E_int| <= 1 |E_0|`. A fusão
    injeta legitimamente energia neste modelo, e um único evento move `0.034 |E_0|`: uma cota de
    sinal reprovaria uma implementação correta. O critério honesto é "o livro de colisões excedeu a
    energia de ligação do sistema".
40. **A implementação está vindicada; a predição falhou; o modelo estava defeituoso.** Os três
    vereditos são distintos e não devem ser confundidos (Seção 4.13.2). Massa exata a `2.4e-16` em
    `12601` passos, momento exato, sem `NaN` — os itens do PISO que protegem essas grandezas
    funcionaram. **Errar uma predição é barato; não registrar por que se errou é caro.**

### Decisões da revisão (c) — a decisão de parar

41. **O runaway é ACEITO como resultado do modelo, e não será mais combatido.** Segunda execução:
    `max m_i = 321.26 m_bar`, `max m_i/M_real = 0.3213`, com `x` já sem qualquer dependência de
    massa. **[M]** Existe um segundo mecanismo, e ele não está no mapa.
42. **A fragmentação NÃO é um canal de contenção; é um canal de concentração de massa.** Ela é
    `2 -> 2` e conserva `m_i + m_j` exatamente, logo **não retira massa do conjunto — só
    redistribui** — e **eleva** o máximo sempre que o parceiro passa de `(1-k)/k = 42.9%` do corpo
    grande (e de `11.1%` quando `f` sai em `0.9`). **[T]** **Este modelo não tem sumidouro de massa
    e, portanto, não tem teto de massa, para nenhum valor de nenhum parâmetro.**
43. **O mecanismo foi decidido por CONTAGEM, não por hipótese.** `226` fusões na execução inteira,
    contra `320` necessárias numa árvore binária para montar `321 m_bar`. `320 > 226`: a fusão não
    podia ter construído o corpo. **[T]** A hipótese da captura dinâmica (`|u|` baixo perto do corpo
    massudo) fica registrada como **não verificada e não necessária**: mesmo com `p_fus = 1` nas
    `226` fusões, a conta não fecha.
44. **`m*` da Seção 4.12 está em unidades do PARCEIRO, não de `m_bar`** — terceira aparição do mesmo
    erro de método: um ponto fixo calculado com um coeficiente co-evolutivo mantido fixo (na revisão
    (b) era `x`; aqui é `m'`). Com as frações medidas, `m* = 2.69 m'`, que não contém nada quando
    `m'` também cresce. **O "teto fechado" está retratado.**
45. **`INV-23(c)` foi RETIRADO da condição de cota, não elevado.** Três formulações, três
    reprovações; uma quarta elevação seria o ajuste post-hoc que este documento proíbe. `|E_int|` é
    a **assinatura quantitativa do runaway aceito**, e um número que mede fielmente um resultado
    aceito não pode reprovar esse resultado. Passa a ser **reportado com sinal**. Mudou-se o **tipo**
    do critério por argumento, não o seu **valor** para caber o número — e é essa distinção que
    separa a decisão de um ajuste de resultado.
46. **A decisão de parar tem justificativa de PRODUTO, e é ela que decide.** Na rodada 1 a transição
    tomava `1.2%` da execução e lia como *glitch*; agora toma `48%` e lê como **processo** — um
    objeto crescendo e comendo o núcleo ao longo de metade da simulação, que é um *runaway merger*
    reconhecível. `774` corpos sobrevivem. O critério declarado pelo usuário ("não sempre, não rápido
    demais") está satisfeito, e a alternativa contida — uma população quase uniforme de corpos entre
    `1` e `5 m_bar` — seria **menos interessante de ver**.
47. **O alvo era da análise, não do modelo.** `max m ~ 3 m_bar` nunca foi propriedade deste modelo;
    era propriedade de uma aproximação de campo médio que falhou três vezes seguidas. Duas rodadas
    de correção encontraram dois mecanismos reais, cada um visível só depois de eliminado o anterior
    — assinatura de que uma terceira rodada encontraria um terceiro. **Errar predição é barato;
    seguir mexendo no modelo para salvar uma predição é caro.**
48. **A predição vigente descreve o runaway em vez de negá-lo** (Seção 4.13.6), continua
    falsificável, e **FOI TESTADA E BATEU** (Seção 4.13.7, 2026-08-08) **[M]**. O ensemble
    `K_SEEDS = 4` confirma todas as linhas: `N_final ∈ [700,900]` (`774`, `783`, `798`, `785`),
    `t_runaway ∈ [1.2, 2.0]` (`1.95`–`1.99`), `max m/M_real ∈ [0.15,0.60]` (`0.306 ± 0.012`),
    duração `> 0.5 t_ff`, `|E_int|/|E_0| ∈ [3,40]`, canais cada `>= 5%`. **A decisão de aceitar o
    runaway deixa de repousar sobre uma execução.** Três ressalvas, todas em 4.13.7 e nenhuma
    decorativa: o ensemble varia **só a semente de colisão** e não testa robustez entre
    **realizações**; `t_runaway` passa colado no teto da banda, e os valores `~1.55` e `48%` que
    este documento publica para a mesma execução **divergem** do ensemble e estão **não
    confirmados**; e uma das quatro execuções terminou com `|p| = nan`, não reproduziu, e está
    excluída por regra pré-declarada — **era justamente a que teria falsificado a predição**, o que
    obriga a exclusão a ser lida com a justificativa completa, não aceita de passagem.

### Decisões da revisão (d) — reconciliação (Seção 9.8)

**A revisão (d) não decide física nova.** Os três itens abaixo são os únicos com conteúdo
normativo; o resto dela é varredura de sobrevivências das revisões anteriores, tabulada em 9.8.

49. **`integrate()` ganha `collision_rng`, e o motivo é a razão de o parâmetro não poder sumir.**
    O fluxo de colisão tem de sobreviver **entre** chamadas de `integrate`. Um gerador recriado a
    cada chamada faz um chamador que avança em pedaços — o visualizador, um quadro por chamada —
    reiniciar o fluxo a cada pedaço: o sorteio de canal fica periódico com o período do quadro e a
    mesma semente passa a dar resultados diferentes conforme o fatiamento, violando `INV-19(c)`.
    **`INV-32` continuava passando**, porque é enunciado por passe — a violação é invisível à suíte
    atual. Omitido, o parâmetro reproduz o comportamento anterior bit a bit. Seção 9.1.2.
50. **Cotas normalizadas por `|E_0|` ganham regime de validade declarado, sem mudar de valor.**
    `TOL-EVENT-CONS` e `TOL-EINT-DRIFT` foram derivadas do resíduo de terceiros medido em pares de
    massa comparável (`1.2%` do termo mútuo **[M]**). O termo mútuo escala com `m_i m_j`, logo com
    um corpo de `321 m_bar` a normalização por `|E_0|` perde sentido. **Valem enquanto
    `max_i m_i / M_real < 0.10`; além disso são reportadas.** A forma escala-invariante
    (razão ao termo mútuo do próprio evento) está **[A]**, com a medição que a decide nomeada.
    Isto é o mesmo tratamento de `INV-23(c)`, `TOL-COURANT` e `TOL-REJECT`: **muda-se o tipo do
    critério onde a derivação acabou, nunca o seu valor para caber o número observado.**
51. **Retratar uma decisão é varredura, não parágrafo.** Dezessete sobrevivências foram encontradas
    nesta revisão, todas do mesmo tipo: a conclusão foi retratada e a frase que a usava como
    premissa ficou. A pior delas — `INV-23(c)` vivo na Seção 6 depois de retirado na Seção 4.10 —
    teria produzido um teste vermelho contra código correto, e o reflexo seguinte teria sido mexer
    no código. **Toda revisão que retrate algo é obrigada a procurar todos os lugares onde o número
    ou o nome retratado aparece**, e não apenas a corrigir onde a decisão foi tomada.
