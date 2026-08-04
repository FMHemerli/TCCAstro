# Glossário Técnico

Definições de termos e conceitos utilizados no projeto, divididos em Física e Computação.

## Física

**Amolecimento de Plummer**

Regularização aplicada ao cálculo da força gravitacional que remove a singularidade do potencial newtoniano em separação nula. A aceleração é calculada com $a_i = G \sum_{j \neq i} m_j (r_j - r_i) / (|r_j - r_i|^2 + \varepsilon^2)^{3/2}$, onde $\varepsilon$ é o parâmetro de amolecimento. O núcleo de par correspondente, $-G m_i m_j / (d^2 + \varepsilon^2)^{1/2}$, é exatamente o potencial de uma esfera de Plummer de escala $\varepsilon$: cada partícula é tratada como massa pontual movendo-se no campo de uma esfera de Plummer centrada em cada uma das outras. Nomeado após o trabalho de Plummer (1911) sobre distribuição de estrelas em aglomerados globulares.

Duas consequências que precisam ser ditas. Primeira: a força de par passa a ser limitada, com máximo em $d = \varepsilon/\sqrt{2}$, e a frequência de par também — é isso que torna defensável integrar com passo de tempo fixo. Segunda: para $d \lesssim \varepsilon$ o sistema simulado não é o problema newtoniano de massas pontuais, e nenhuma estrutura abaixo dessa escala é resolvida. Por isso $\varepsilon$ é um parâmetro do modelo, não um parâmetro de ajuste numérico, e deve constar de todo resultado publicado. O amolecimento não substitui a exclusão do termo $i = j$, que é feita explicitamente no cálculo.

**Colapso frio**

Configuração inicial de um sistema gravitacional em que as partículas começam em repouso (velocidades nulas) e caem sob a ação mútua da força gravitacional. É uma condição inicial simples que permite testar a conservação de propriedades dinâmicas (energia, quantidade de movimento) e comparar a evolução medida contra previsões analíticas, como o tempo de queda livre.

**Deriva secular**

Variação lenta e sistemática de uma quantidade conservada (como energia) ao longo de uma integração numérica, causada por erro de truncamento do método de integração. Diferencia-se de oscilações limitadas em torno de um valor fixo: na deriva, o erro acumula de forma sistemática e cresce com o tempo decorrido, ainda que possa haver oscilação sobreposta. Integradores simpléticos não exibem deriva secular, apenas oscilação limitada; entre os métodos não simpléticos a deriva é o comportamento típico, e é o que se observa nos dois métodos não simpléticos deste projeto (Euler explícito e Runge-Kutta de 4ª ordem).

**Energia potencial gravitacional**

Energia armazenada na configuração espacial das partículas interagindo pela gravidade, dada por $U = -\sum_{i \lt j} G m_i m_j / (|r_i - r_j|^2 + \varepsilon^2)^{1/2}$ com amolecimento de Plummer. Em um sistema isolado com partida fria (velocidades nulas) tem-se $K_0 = 0$ e portanto $E_0 = U_0$ **no passo inicial**. Como o sistema amolecido é hamiltoniano, a energia total é conservada no fluxo contínuo, logo $E(t) = U_0$ em todo instante; todo desvio dessa identidade é erro numérico. Note que isso **não** significa $E(t) = U(t)$ para $t > 0$: assim que o colapso começa, $K > 0$ e $U \lt U_0$.

**Euler semi-implícito**

Método de integração numérica de equações de movimento que atualiza a velocidade primeiro e então a posição usando a velocidade já atualizada: $v_{t+1} = v_t + a_t \cdot dt$ e $r_{t+1} = r_t + v_{t+1} \cdot dt$. Também chamado Euler–Cromer. Ordem de convergência 1 em posição (erro local $O(dt^2)$, erro global $O(dt)$). Apesar da ordem baixa, **é simplético** — o mapa é a composição de dois cisalhamentos, cada um com jacobiano unitário —, e é essa propriedade, não a ordem, que explica seu comportamento energético muito melhor que o de Euler explícito: a energia oscila com amplitude $O(dt)$ e não deriva. Conserva também o momento angular total exatamente, a menos de arredondamento. Usado no trabalho de 2019 como integrador de referência.

**Integrador simplético**

Método de integração que preserva a estrutura simplética das equações de Hamilton, uma propriedade geométrica que governa sistemas conservativos. A razão do bom comportamento energético é que um integrador simplético de ordem $p$ e passo $h$ conserva exatamente um hamiltoniano sombra $H_h = H + O(h^p)$: a energia medida oscila com amplitude $O(h^p)$ em torno de um valor ligeiramente deslocado da energia verdadeira, e não deriva. A garantia é assintótica, válida enquanto $h$ for pequeno frente a todas as escalas de tempo do sistema, não incondicional. Neste projeto são simpléticos o Euler semi-implícito e o Verlet de velocidades; não são o Euler explícito e o Runge-Kutta de 4ª ordem. A importância surge em simulações de horizonte longo, onde deriva leva a resultados qualitativamente incorretos.

**Ordem de convergência**

Taxa na qual o erro numérico de um integrador diminui com redução do tamanho do passo: um integrador de ordem $p$ tem erro local $O(dt^{p+1})$ e erro global $O(dt^p)$. Medida empiricamente pelo refinamento do tamanho de passo em comparação com uma solução de referência e ajuste de lei de potência. Permite classificar e comparar métodos independentemente da implementação específica.

**Problema de n-corpos**

Conjunto de N partículas pontuais com massa, interagindo mutuamente através de uma força (aqui, gravitacional) que depende de suas posições. A simulação integra as equações de movimento ao longo do tempo. É um problema central em astrofísica computacional, com aplicações em dinâmica de galáxias e aglomerados estelares. Custo computacional é $O(N^2)$ por passo de tempo para cálculo direto da força.

**Runge-Kutta de 4ª ordem**

Método de integração que avalia a aceleração em quatro estágios intermediários do passo de tempo para alcançar ordem de convergência 4. Custa quatro avaliações de força por passo, contra uma dos métodos de Euler e do Verlet. Formulado por Runge e Kutta na virada do século XIX para o XX; tratamento moderno em [HAIRER 1993]. Produz erros substancialmente menores que métodos de ordem 1 ou 2 no mesmo tamanho de passo, mas **não é simplético**: a energia deriva secularmente, ainda que a partir de um patamar baixo.

**Sistema isolado**

Sistema cuja energia total é conservada porque não há troca de energia com o exterior. A simulação implementa um sistema gravitacional isolado: sem damping, sem fontes externas, sem radiação. A identidade $E(t) = E_0$ (energia total conservada) é esperada no fluxo contínuo; desvios quantificam erro do integrador.

**Tempo de queda livre**

Tempo que uma esfera de densidade uniforme $\rho$, inicialmente em repouso e sem pressão, leva para colapsar até raio nulo sob a própria gravidade: $t_{ff} = \sqrt{3\pi / (32 G \rho)}$. O resultado independe do raio inicial da camada considerada — todas as camadas chegam ao centro simultaneamente —, e é isso que faz de $t_{ff}$ uma previsão fechada contra a qual medir a simulação. Usado como unidade de tempo adimensional para colocar simulações de diferentes tamanhos e densidades na mesma escala; permite comparações e validações contra previsões analíticas.

**Verlet de velocidades**

Método de integração simplético de ordem 2 que evolui ambas as posições e velocidades de forma a conservar a estrutura geométrica das equações de Hamilton: $v_{t+1/2} = v_t + (a_t dt)/2$, $r_{t+1} = r_t + v_{t+1/2} dt$, $v_{t+1} = v_{t+1/2} + (a_{t+1} dt)/2$. Ordem 2 permite passos de tempo maiores que Euler com igual precisão, mas exige menos trabalho que RK4. Introduzido por Verlet (1967).

---

## Computação

**Avaliação de força**

Cálculo da aceleração (ou força por unidade de massa) de cada partícula devida a todas as outras, somando as contribuições par a par. Em uma simulação de n-corpos direto, requer $O(N^2)$ operações por passo. Em códigos de benchmark, o tempo desta operação domina o tempo total de integração, tornando-a o critério principal para comparar eficiência de implementação.

**Backend**

Camada de implementação de uma função numérica (aqui, avaliação de força). O projeto fornece seis backends: Python puro, PyTorch eager em CPU, PyTorch compilado em CPU, PyTorch eager em GPU, PyTorch compilado em GPU, e kernel Triton manual. Cada backend implementa a mesma função matemática mas com diferentes trade-offs entre simplicidade, portabilidade e desempenho.

**Compilação JIT**

Just-In-Time: compilação de código durante a execução, não antes. Numba (2015) compila funções Python anotadas para código de máquina LLVM; PyTorch 2 com `torch.compile` (Ansel et al., 2024) captura o grafo de computação Python e o compila para código otimizado. Permite combinar flexibilidade de linguagens de alto nível com desempenho próximo a C.

**CUDA**

Compute Unified Device Architecture, pilha de programação de NVIDIA para paralelismo em GPUs NVIDIA. Permite escrever kernels customizados em C/C++. Usado no trabalho de 2019 através do Numba [LAM 2015], que compila código Python anotado para CUDA. Nota: o projeto atual usa ROCm, não CUDA.

**Eager**

Modo de execução em que operações são avaliadas imediatamente quando invocadas, não diferidas em um grafo. PyTorch eager mode é o padrão; cada operação executa logo, permitindo debugging interativo. Contrasta com modo compilado (graph-based), onde operações são registradas para otimização global antes de execução.

**GPU**

Graphics Processing Unit. Acelerador paralelo com muitos núcleos de baixa potência, otimizado para cargas de trabalho paralelas massivas. Ideal para computação científica quando o algoritmo explora paralelismo de dados. No contexto deste projeto: NVIDIA Tesla T4 (2019) e AMD Radeon RX 9060 XT (2026).

**Kernel**

Função executada em paralelo por muitos threads em uma GPU (ou CPU com paralelismo), operando sobre blocos de dados. Um kernel Triton é escrito em linguagem de alto nível e compilado para código de GPU, abstraindo os detalhes de sincronização entre threads. Permite expressar algoritmos paralelos de forma mais concisa que CUDA C bruto.

**Mediana e intervalo interquartil**

Estatísticas de localização e dispersão robustas a outliers. A mediana é o valor central (50º percentil) de um conjunto de medições. O intervalo interquartil (IQR) é a distância entre 25º e 75º percentis, capturando a largura da metade central dos dados. Usado em vez de média e desvio padrão em benchmarks para ignorar picos ocasionais de latência causados por escalonador do SO.

**O(N²)**

Notação de ordem de magnitude: um algoritmo cuja complexidade cresce como o quadrado do tamanho da entrada. Avaliação direta de força em n-corpos é O(N²) porque requer somar sobre todos os N pares de partículas. Inclinação log-log de 2 em uma varredura de N confirma comportamento O(N²).

**Paralelismo**

Execução simultânea de múltiplas operações. Paralelismo de dados: muitos processadores aplicam a mesma operação a dados diferentes (SIMD, GPUs). Paralelismo de tarefa: múltiplos processadores executam tarefas diferentes (multiprocessamento, OpenMP). O trabalho de 2019 explorou ambos via Numba [LAM 2015]; a modernização de 2026 os explora via PyTorch [ANSEL 2024], com paralelismo de dados na GPU e paralelismo de threads na CPU.

**Precisão simples e dupla**

Float32 (precisão simples) e float64 (precisão dupla) são representações padrão IEEE 754. Float32 ocupa 4 bytes e oferece ~7 dígitos decimais; float64 oferece ~16 dígitos. GPUs de consumo têm desempenho muito mais lento em double que single. O projeto mede ambas as precisões para avaliar trade-offs.

**ROCm/HIP**

ROCm (Radeon Open Compute) é a pilha open-source de AMD para programação de GPUs. HIP (Heterogeneous-Compute Interface for Portability) é uma linguagem e API que abstrai detalhes de CUDA e ROCm, permitindo escrever código portável entre GPUs NVIDIA e AMD. Usado neste projeto para suportar AMD Radeon RX 9060 XT.

**Tiling**

Particionamento de um array multidimensional em blocos (tiles) que cabem em cache ou memória compartilhada de um acelerador. Permite computar uma operação grande (como $O(N^2)$) fazendo múltiplas operações pequenas, economizando memória. No cálculo de força de n-corpos, tiling sobre partículas i reduz a materialização de um tensor N × N × 3 para N × B × 3, onde B é tamanho do bloco.

**Triton**

Linguagem e compilador para kernels GPU de alto nível, abstrai detalhes de sincronização de threads e permite expressar kernels estruturados em blocos sem lidar com CUDA C. Compila para LLVM IR e daí para código CUDA ou ROCm HIP. Usado para escrever o kernel manual (degrau 6) do projeto.

**Warmup**

Execução inicial de um kernel ou operação para aquecê-lo (compilar, carregar em cache, estabilizar frequência de clock). Medições de tempo nunca começam no primeiro call; descartam as primeiras execuções para evitar outliers de inicialização. Crítico em benchmarks GPU, onde overhead de inicialização pode mascarar desempenho real.
