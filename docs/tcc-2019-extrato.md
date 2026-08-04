# Extrato da monografia de 2019

Fonte: `TCC_final_cutv3.pdf` — "Computação de Alto Desempenho Usando Numba para Cálculo de
Agregação de Matéria", Flávio Manoel Santos Hemerli, UFES, 2019. Orientador: Prof. Dr. Roberto
Colistete Júnior. Banca: Prof. Dr. Mario Alberto Simonato Altoé, Prof. Msc. Gabriel Lessa da Silva
Lavagnoli. Aprovado, nota 9/10.

Este documento existe para fixar o que a monografia realmente afirma, com a seção de origem, de modo
que a modernização de 2026 não contradiga nem exagere o trabalho original.

## Critério de erro (§3.4, p. 19)

Literal: "utilizamos a conservação da energia total para calcular o erro da simulação (E/E0)".
Referência citada: DANIEL; FOSTER-O'NEAL, 2012.

Complemento em §3.5 (p. 23): "Como o modelo se trata de uma simulação n-corpos de colapso frio,
todas as partículas do sistema iniciam a simulação com velocidade nula, incorrendo em uma energia
total igual a soma de todas as energias potenciais."

**Tolerância declarada: 5%**, explicitamente arbitrária — "esperou-se que a diferença entre elas,
isto é, o erro, estivesse dentro de um limite tolerável, estabelecido arbitrariamente como sendo 5%".

## Justificativa documentada para Verlet e Runge-Kutta (§3.4, p. 19)

Literal: "Alguns métodos (como o método de Euler-Cromer, Verlet e Runge-Kutta de diferentes ordens)
podem reduzir o erro enquanto aumentam o custo computacional, outros podem causar comportamento
anômalo no sistema tornando a modelagem sem sentido (GIORDANO; NAKANISHI, 2006)."

A intenção de testar Verlet e RK4 está registrada na monografia, com referência. Não é reconstrução
posterior.

## Por que Euler foi escolhido (§3.4, p. 19–20)

Dois motivos, ambos declarados:

1. "Como a simplicidade foi definida como um fator importante para a pesquisa, optou-se por utilizar
   o Método de Euler."
2. O custo extra do Euler era **útil** para o objetivo do trabalho: "o efeito colateral do aumento de
   custo computacional pode ser ignorado uma vez que um dos objetivos da pesquisa é a comparação de
   desempenho, permitindo que tal aumento seja usado a favor do estudo ao aumentar a diferença de
   tempo de execução entre métodos mais eficientes e menos eficientes."

Ou seja: a escolha do integrador era subordinada ao objetivo de benchmark. Não foi desconhecimento.

## Discrepância entre o Euler descrito e o Euler implementado

O texto (§3.4, p. 19) escreve o método como:

    v_{t+1} = v_t + a_t dt
    r_{t+1} = r_t + v_t dt

que é Euler **explícito** (posição usa `v_t`). O código em §3.5 (p. 22) atualiza toda a velocidade
primeiro e só depois a posição, num segundo laço — usando portanto `v_{t+1}`, que é Euler
**semi-implícito** (Euler-Cromer). O implementado é o simplético; o descrito não é.

Consequência para 2026: o degrau "Euler do TCC" corresponde a `symplectic_euler`, não a `euler`.

## Força e potencial (§3.5, p. 22 e 24)

Aceleração, do código: `dsq = dx²+dy²+dz² + sft²`, `gA = G*m[j]/(dsq*sqrt(dsq))`, `a += gA*d`.
Isto é `a_i = G m_j (r_j - r_i) / (|r|² + eps²)^{3/2}` — **Plummer**.

Potencial, do código: `U += -G*m[i]*m[j]/sqrt(|r|² + sft²)` — **potencial de Plummer**, consistente
com a força.

O par força/potencial de 2019 já era consistente. `E(t) = U_0` é exatamente verdadeiro no modelo de
2019, não aproximadamente. A implementação de 2026 usa o mesmo par.

## Parâmetros

| grandeza | valor | origem |
|---|---|---|
| `eps` (atenuação) | `1e-10` | §3.5 p. 23, "valor arbitrário" da etapa inicial |
| `dt` | `0.01` | §3.5 p. 24, após ajuste empírico; `1e-5` no início do desenvolvimento |
| `G` | `6.67408e-11` | §3.6; valor `1` na fase de desenvolvimento |
| massa | `1e9 kg` | §3.6; valor `1` na fase de desenvolvimento |
| distribuição | cúbica homogênea | §3.5 p. 24 — escolhida por ser reprodutível, ao contrário da aleatória |
| N (validação) | 1000 | §3.6 |
| N (benchmark) | 32768 | §3.6 |

## Separação entre validação e benchmark (§3.6, p. 28) — corrige a auditoria

Literal: "Então decidiu-se por alterar o número de corpos de 1.000 para 32.768 e executar o software
para uma interação apenas, **omitindo os cálculos de energia**."

A validação por energia foi feita em N = 1000, com tolerância de 5%, durante o desenvolvimento. O
benchmark em N = 32768 desligou a energia **de propósito**, e a monografia diz isso.

Isto corrige o achado 1 da auditoria de 2026 ("os seis kernels nunca foram verificados"). A forma
correta é mais estreita: a validação por energia existiu, num regime distinto do cronometrado, e não
há teste de equivalência **entre implementações**. O `flag_Energy=False` nos notebooks é a execução
da decisão documentada, não um descuido.

## Precisão dupla na GPU foi imposta, não escolhida (§3.5, p. 25)

Literal: "Por limitações da ferramenta foi necessário usar precisão dupla para os valores de ponto
flutuante, o que incorre em um menor desempenho e maior precisão."

Relevante para 2026: a comparação fp32/fp64 não é uma variação livre sobre o trabalho original — o
fp64 do TCC era imposição do Numba CUDA daquela versão.

## Dados da Tabela 1 (p. 30) ausentes dos notebooks

A monografia reporta, no Dell G3, dois pontos que não existem em `results/2019/benchmarks.csv`
porque nunca estiveram nos notebooks:

| ferramenta | Dell G3 (s) |
|---|---|
| C | 7,8844 |
| C OpenMP (paralelo) | 1,3900 |

E separa "Numba GPU //" de "Numba GPU // kernel" (Colab 0,2865 / 0,2828; Dell G3 0,7350 / 0,7333),
isto é, tempo total contra tempo de kernel.

## Tabela 4 (p. 33) — teste `mu_0` de Cosmologia Observacional

Cálculo não relacionado ao n-corpos, incluído para comparação mais ampla. Contém explicitamente
precisão simples contra dupla:

| método | Dell G3 (s) | Google Colab (s) |
|---|---|---|
| CUDA SP | 0,00128 | 0,00156 |
| CUDA DP | 0,03770 | 0,00488 |
| Numba GPU SP | 0,04930 | 0,00833 |
| Numba GPU DP | 0,06350 | 0,00693 |

No Dell G3 (GTX 1050 Ti) a razão CUDA DP/SP é ~29x. O trabalho de 2019 já tinha medido a penalidade
de fp64 em GPU de consumo, em outro problema.

## Trabalho futuro nomeado pela própria monografia (§5, p. 36)

Literal: "é possível aumentar a complexidade do modelo criado para simular um sistema mais realista,
alterando o método de aproximação numérica e a distribuição de cargas."

Trocar o método de aproximação numérica e trocar a distribuição são exatamente o que a modernização
de 2026 faz. O mandato está no documento original.

## Escopo declarado, a ser respeitado (§4, p. 34)

Literal: "o problema de n-corpos serve o propósito de teste de desempenho do método como aplicação
em Física para um caso de grande complexidade. Ele não representa uma medida genérica de desempenho
para qualquer aplicação."

O objeto do trabalho é desempenho entre formas de implementação. Corretude física não era a
pergunta, e a ausência de validação robusta segue disso — é limite de escopo declarado, não falha.
