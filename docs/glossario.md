# Glossário Técnico

Definições de termos e conceitos utilizados no projeto, divididos em Física e Computação.

## Física

**Amolecimento de Plummer**

Regularização aplicada ao cálculo da força gravitacional que remove a singularidade do potencial newtoniano em separação nula. A aceleração é calculada com $a_i = G \sum_{j \neq i} m_j (r_j - r_i) / (|r_j - r_i|^2 + \varepsilon^2)^{3/2}$, onde $\varepsilon$ é o parâmetro de amolecimento. O núcleo de par correspondente, $-G m_i m_j / (d^2 + \varepsilon^2)^{1/2}$, é exatamente o potencial de uma esfera de Plummer de escala $\varepsilon$: cada partícula é tratada como massa pontual movendo-se no campo de uma esfera de Plummer centrada em cada uma das outras. Nomeado após o trabalho de Plummer (1911) sobre distribuição de estrelas em aglomerados globulares.

Duas consequências que precisam ser ditas. Primeira: a força de par passa a ser limitada, com máximo em $d = \varepsilon/\sqrt{2}$, e a frequência de par também — é isso que torna defensável integrar com passo de tempo fixo. Segunda: para $d \lesssim \varepsilon$ o sistema simulado não é o problema newtoniano de massas pontuais, e nenhuma estrutura abaixo dessa escala é resolvida. Por isso $\varepsilon$ é um parâmetro do modelo, não um parâmetro de ajuste numérico, e deve constar de todo resultado publicado. O amolecimento não substitui a exclusão do termo $i = j$, que é feita explicitamente no cálculo.

**Ressalva obrigatória quando as colisões estão ligadas.** Com $\chi = 0.1$ o raio de contato somado de um par típico vale $R_i + R_j = 0.01\ \mathrm{m}$, **cinco vezes menor** que $\varepsilon = 0.05\ \mathrm{m}$. Portanto **as colisões ocorrem inteiramente dentro da região regularizada**: a velocidade de impacto e a profundidade do poço de par no contato são as do potencial de Plummer, não as de massas pontuais. A diferença não é cosmética — a profundidade no contato é $G m_i m_j / \sqrt{(R_i+R_j)^2 + \varepsilon^2}$ e não $G m_i m_j / (R_i+R_j)$, um fator $5.1$ para o par de massa média. Isso não é um erro, é a semântica correta do modelo, mas tem de constar da prosa. O regime alternativo, $R_i + R_j > \varepsilon$, em que o amolecimento nunca atua, foi examinado e **vetado**: ele produz $\sim 85$ colisões por partícula por rebote e troca uma imprecisão em $d \lesssim \varepsilon$ por uma mudança de regime do sistema inteiro. Ver `docs/simulacao-estocastica.md`, Seção 4.2.

**Colapso frio**

Configuração inicial de um sistema gravitacional em que as partículas começam em repouso (velocidades nulas) e caem sob a ação mútua da força gravitacional. É uma condição inicial simples que permite testar a conservação de propriedades dinâmicas (energia, quantidade de movimento) e comparar a evolução medida contra previsões analíticas, como o tempo de queda livre.

**Deriva secular**

Variação lenta e sistemática de uma quantidade conservada (como energia) ao longo de uma integração numérica, causada por erro de truncamento do método de integração. Diferencia-se de oscilações limitadas em torno de um valor fixo: na deriva, o erro acumula de forma sistemática e cresce com o tempo decorrido, ainda que possa haver oscilação sobreposta. Integradores simpléticos não exibem deriva secular, apenas oscilação limitada; entre os métodos não simpléticos a deriva é o comportamento típico, e é o que se observa nos dois métodos não simpléticos deste projeto (Euler explícito e Runge-Kutta de 4ª ordem).

**Detecção varrida**

Teste de contato que examina o **segmento inteiro** percorrido por um par durante um passo, em vez de apenas as configurações de início e fim. Com $\Delta r = r_j - r_i$ e $\Delta v = v_j - v_i$ constantes ao longo do passo, a separação ao quadrado $|\Delta r + t\,\Delta v|^2$ é uma parábola convexa em $t$, cujo mínimo sobre $[0, h]$ ocorre em $t^{*} = \mathrm{clamp}(-(\Delta r \cdot \Delta v)/|\Delta v|^2,\ 0,\ h)$; há colisão se $|\Delta r + t^{*}\Delta v| < R_i + R_j$. Duas guardas são normativas: $|\Delta v|^2 = 0$ adota $t^{*} = 0$, e o par só é candidato se $\Delta r \cdot \Delta v < 0$ **no início** do passo, o que impede que um par recém-colidido dispare de novo enquanto ainda se separa.

A hipótese de movimento retilíneo **não é uma aproximação** no esquema adotado: no Verlet KDK a posição avança por $r^{n+1} = r^n + h\,v^{n+1/2}$ com a velocidade de meio passo constante ao longo do subpasso, de modo que o varrido descreve exatamente a trajetória que o integrador percorre — desde que $\Delta v$ seja tomado das velocidades de **meio passo**, e é por isso que a colisão é inserida *dentro* do *drift*. Consequência central: **o detector varrido não pode perder um contato, para nenhum valor do número de Courant colisional**, porque $\Delta r \cdot \Delta v$ é monotonicamente não decrescente ao longo do passo e a aproximação máxima está sempre dentro de um passo cuja guarda passa. Um encontro frontal é o caso **mais fácil** para ele, não o mais difícil; o caso marginal é o rasante.

**Energia interna acumulada**

Acumulador $E_{\rm int}$ que registra a energia mecânica convertida em graus de liberdade internos pelos desfechos de colisão, de modo que a quantidade conservada do sistema com colisões passe a ser $E_{\rm total} = K + U + E_{\rm int}$ em lugar de $K + U$. A convenção de sinal é $E_{\rm int} \mathrel{+}= -(\Delta K + \Delta U)$: colisão elástica não contribui, fusão contribui com $T_{\rm cm} - E_{\rm grav}$, fragmentação com $E_{\rm grav}(\text{depois}) - E_{\rm grav}(\text{antes})$.

**O ponto de avaliação é o que dá sentido à grandeza, e é normativo:** os acumuladores são avaliados **através do mapa de desfecho, com as posições congeladas em $t^{*}$**, e nunca ao longo do passo de integração. Se fossem avaliados ao longo do passo, a contabilidade absorveria também o erro de truncamento do integrador, $E_{\rm total}$ passaria a ser conservada trivialmente e o diagnóstico-chefe do projeto — separar simpléticos de não simpléticos pelo comportamento da energia — seria destruído pela própria máquina construída para preservá-lo. Acumula-se sempre em precisão dupla, mesmo com o núcleo em fp32, porque é uma soma de milhares de incrementos de sinais opostos. $|E_{\rm int}|$ é **reportado, não limitado**: ele cresce com a massa do corpo dominante, que neste modelo não tem teto, e valores muito acima de $|E_0|$ são a assinatura do runaway aceito, não um defeito.

**Energia potencial gravitacional**

Energia armazenada na configuração espacial das partículas interagindo pela gravidade, dada por $U = -\sum_{i \lt j} G m_i m_j / (|r_i - r_j|^2 + \varepsilon^2)^{1/2}$ com amolecimento de Plummer. Em um sistema isolado com partida fria (velocidades nulas) tem-se $K_0 = 0$ e portanto $E_0 = U_0$ **no passo inicial**. Como o sistema amolecido é hamiltoniano, a energia total é conservada no fluxo contínuo, logo $E(t) = U_0$ em todo instante; todo desvio dessa identidade é erro numérico. Note que isso **não** significa $E(t) = U(t)$ para $t > 0$: assim que o colapso começa, $K > 0$ e $U \lt U_0$.

**Euler semi-implícito**

Método de integração numérica de equações de movimento que atualiza a velocidade primeiro e então a posição usando a velocidade já atualizada: $v_{t+1} = v_t + a_t \cdot dt$ e $r_{t+1} = r_t + v_{t+1} \cdot dt$. Também chamado Euler–Cromer. Ordem de convergência 1 em posição (erro local $O(dt^2)$, erro global $O(dt)$). Apesar da ordem baixa, **é simplético** — o mapa é a composição de dois cisalhamentos, cada um com jacobiano unitário —, e é essa propriedade, não a ordem, que explica seu comportamento energético muito melhor que o de Euler explícito: a energia oscila com amplitude $O(dt)$ e não deriva. Conserva também o momento angular total exatamente, a menos de arredondamento. Usado no trabalho de 2019 como integrador de referência.

**Integrador simplético**

Método de integração que preserva a estrutura simplética das equações de Hamilton, uma propriedade geométrica que governa sistemas conservativos. A razão do bom comportamento energético é que um integrador simplético de ordem $p$ e passo $h$ conserva exatamente um hamiltoniano sombra $H_h = H + O(h^p)$: a energia medida oscila com amplitude $O(h^p)$ em torno de um valor ligeiramente deslocado da energia verdadeira, e não deriva. A garantia é assintótica, válida enquanto $h$ for pequeno frente a todas as escalas de tempo do sistema, não incondicional. Neste projeto são simpléticos o Euler semi-implícito e o Verlet de velocidades; não são o Euler explícito e o Runge-Kutta de 4ª ordem. A importância surge em simulações de horizonte longo, onde deriva leva a resultados qualitativamente incorretos.

**Lei de potência truncada**

Distribuição usada para o espectro de massas: $p(m) = C\,m^{-\alpha}$ em $[m_{\min}, m_{\max}]$, com $C = (1-\alpha)/(m_{\max}^{1-\alpha} - m_{\min}^{1-\alpha})$ para $\alpha \neq 1$ e $p(m) = 1/(m \ln(m_{\max}/m_{\min}))$ no caso degenerado $\alpha = 1$, em que a normalização acima é $0/0$. O padrão do projeto é $\alpha = 2.35$ (Salpeter) com razão de truncamento $m_{\max}/m_{\min} = 1000$ fixada. A amostragem é por CDF inversa, exata e sem rejeição.

O truncamento não é um detalhe de implementação: sem ele a lei de Salpeter tem massa total divergente na extremidade inferior e a massa média não existe em forma fechada. Com ele, $m_{\min}$ fica **determinado** pela massa média alvo em vez de escolhido — é raiz de uma equação fechada, não um parâmetro livre.

**Maxwelliana truncada**

Distribuição de velocidades iniciais do projeto: densidade isotrópica no **espaço de velocidades** $f(v)\,d^3v \propto \exp(-|v|^2/2\sigma^2)\,d^3v$ para $|v| \le v_{\rm cut}$, e exatamente zero acima, com $v_{\rm cut} = f_{\rm cut}\,v_{\rm esc}$.

**A distinção que esta entrada existe para fixar, e é o ponto mais fácil de enunciar errado em todo o projeto.** Duas funções diferentes descrevem a mesma distribuição, e a monotonicidade vale para uma e não para a outra:

- $f(v) \propto \exp(-|v|^2/2\sigma^2)$, densidade por unidade de **volume** $d^3v$, é **estritamente decrescente** na rapidez, com máximo em $v = 0$;
- $p(|v|) \propto |v|^2 \exp(-|v|^2/2\sigma^2)$, densidade por unidade de **rapidez** $d|v|$, **anula-se** em $|v| = 0$, **cresce** até $|v| = \sqrt{2}\,\sigma$ e só então decresce.

As duas afirmações são simultaneamente corretas porque medem coisas diferentes: $p(s) = 4\pi s^2 f(s)$, e é o fator de área da casca esférica, não a exponencial, que domina em $s$ pequeno. **Enunciados permitidos:** "a densidade no espaço de velocidades é estritamente decrescente na rapidez"; "a rapidez mais provável é $\sqrt{2}\,\sigma$". **Enunciados proibidos:** "partículas rápidas são mais raras que partículas lentas" (falso abaixo de $\sqrt{2}\,\sigma$); "a maioria das partículas está quase parada" (falso: $p(s) \to 0$ quando $s \to 0$); e qualquer frase que não diga **em relação a que medida** a monotonicidade vale.

Trocar a maxwelliana por uma $p(|v|)$ estritamente decrescente foi examinado e rejeitado: exigiria $f$ divergindo como $s^{-2}$ na origem — uma cúspide de densidade infinita em $v = 0$, que não é distribuição de equilíbrio de hamiltoniano algum.

**Número de Courant colisional**

Razão $C_{\rm coll} = |u|\,\Delta t / (R_i + R_j)$ entre o deslocamento relativo de um par em um passo e o diâmetro da sua zona de contato. **É uma grandeza REPORTADA, não uma condição de validade**, e nenhum teste falha nem nenhuma execução é invalidada por ela.

A razão do rebaixamento merece ser registrada porque contraria a intuição. Parece óbvio que $C_{\rm coll} > 1$ significa "o par atravessa a zona de contato dentro de um passo e o evento é perdido"; esse raciocínio pressupõe que a colisão é resolvida **no fim do passo**, o que o esquema adotado não faz. Com a colisão inserida dentro do *drift* e a normal avaliada em $t^{*}$, a detecção varrida não perde contato algum, para nenhum $C_{\rm coll}$. Medido: $C_{\rm coll} = 1.81$ e um valor quatro vezes menor produzem a mesma física dentro de $0.5\%$. **Um número cujo valor e cujo quarto dão o mesmo resultado não é uma condição de validade.**

**Ordem de convergência**

Taxa na qual o erro numérico de um integrador diminui com redução do tamanho do passo: um integrador de ordem $p$ tem erro local $O(dt^{p+1})$ e erro global $O(dt^p)$. Medida empiricamente pelo refinamento do tamanho de passo em comparação com uma solução de referência e ajuste de lei de potência. Permite classificar e comparar métodos independentemente da implementação específica.

**Parâmetro de regime**

Grandeza adimensional $x$ que decide, estatisticamente, o desfecho de uma colisão, comparando a energia do impacto com uma energia de coesão de referência:

$$x := \frac{T_{\rm cm}}{E_{\rm bind}} = \frac{|u|^2}{v_{\rm coh}^2}, \qquad T_{\rm cm} = \tfrac{1}{2}\mu|u|^2, \quad E_{\rm bind} = \tfrac{1}{2}\mu\,v_{\rm coh}^2, \quad v_{\rm coh} = V_{\rm CHAR}$$

A massa reduzida $\mu$ cancela, e com ela **toda** dependência com as massas dos corpos: $x$ depende apenas da velocidade relativa no contato. O mapa $x \mapsto (p_{\rm fus}, p_{\rm el}, p_{\rm frag}) = (1/x,\ 3,\ x)/Z$, com $Z = 1/x + 3 + x$, não tem parâmetro livre algum: soma exatamente $1$ por construção, é estritamente positivo em todo canal e todo $x$ (nenhum desfecho é jamais proibido), $p_{\rm fus}$ decresce e $p_{\rm frag}$ cresce monotonicamente — lento funde, rápido fragmenta —, e é exatamente simétrico sob $x \to 1/x$. O platô elástico é exatamente uma década centrada em $x = 1$, onde $p_{\rm el} = 3/5$.

**Registro de um erro instrutivo:** uma versão anterior incluía em $E_{\rm bind}$ um termo gravitacional, com a justificativa de que sem ele $x \ge 1$ identicamente e a fusão ficaria inalcançável. A justificativa era falsa — no núcleo colapsado a separação local é comparável a $\varepsilon$, o poço de Plummer já é raso ali, e o piso real de $x$ é $\approx 0.4$ e não $1$ — e o termo **causava** o crescimento descontrolado que se acreditava que ele contivesse. Foi retirado.

**Problema de n-corpos**

Conjunto de N partículas pontuais com massa, interagindo mutuamente através de uma força (aqui, gravitacional) que depende de suas posições. A simulação integra as equações de movimento ao longo do tempo. É um problema central em astrofísica computacional, com aplicações em dinâmica de galáxias e aglomerados estelares. Custo computacional é $O(N^2)$ por passo de tempo para cálculo direto da força.

**Raio de contato**

Raio geométrico atribuído a cada corpo para efeito de detecção de colisão, $R_i = R_{\rm ref}\,(m_i/\bar{m})^{1/3}$ com $R_{\rm ref} = \chi\,\varepsilon$. O expoente $1/3$ corresponde a densidade material constante, e $\chi$ é adimensional.

**$\chi$ é irredutivelmente um parâmetro de modelo, e deve constar de todo artefato de saída, como $\varepsilon$.** A densidade nominal do projeto **não** fornece um raio físico: uma esfera de $10^9\ \mathrm{kg}$ a $\rho = 10^9\ \mathrm{kg/m^3}$ teria raio $0.62\ \mathrm{m}$, e duas delas se tocariam à separação interpartícula média — o que revela que essa $\rho$ é a densidade **do sistema**, artefato de contabilidade geométrica herdado da grade inicial, e não uma densidade material.

O valor $\chi = 0.1$ está fixado **por medição**, não por escolha: a grandeza normativa é o número médio de colisões por partícula durante o primeiro rebote, com faixa-alvo $[0.5, 2]$, e $\chi = 0.1$ mede $0.94$, no centro da faixa. Abaixo dela as colisões são raras demais para aparecer; acima, a colisão e não a gravidade passa a governar o núcleo.

**Razão virial**

Quantidade adimensional $Q = 2K/|U|$ que mede quão longe do equilíbrio de virial um sistema autogravitante começa. $Q = 0$ é o colapso frio (partículas em repouso); $Q = 1$ é o equilíbrio de virial, em que não há colapso e o observável-chefe — o instante de cruzamento pelo centro — deixa de existir; $Q = 2$ é o limite de desligamento ($E = 0$). O padrão do projeto para as extensões estocásticas é $Q = 0.25$, escolhido por ficar longe dos dois extremos degenerados, manter o sistema fortemente ligado ($E = (Q/2 - 1)|U| = -0.875|U|$, sem evaporação apreciável no horizonte de $3\,t_{ff}$) e produzir efeito mensurável mas não destrutivo. $Q = 0$ deve reproduzir o colapso frio **bit a bit**, e isso é verificado como invariante.

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

**Passo fixo do fluxo de colisão**

Regra de reprodutibilidade do modelo estocástico de colisão: o fluxo de números aleatórios consome **exatamente dois** valores uniformes por evento aceito, sempre, **independentemente do canal sorteado** — um para escolher o desfecho e outro para o parâmetro do desfecho, retirado mesmo quando o canal não o usa. Um passe com zero eventos consome zero. Se o segundo sorteio fosse condicional ao canal, dois passes com o mesmo número de eventos mas canais diferentes deixariam o gerador em estados distintos, e a execução deixaria de ser reprodutível.

O corolário operacional é que **o gerador pertence ao chamador**, não à função de integração. Enquanto uma execução era uma única chamada, reconstruí-lo internamente a partir de uma semente era indistinguível de um fluxo contínuo; deixou de ser quando o visualizador em tempo real passou a avançar a simulação em pedaços, uma chamada por quadro. Nesse padrão cada chamada reiniciava o fluxo, de modo que o sorteio de canal virava **periódico com o período do quadro** e a mesma semente produzia resultados diferentes conforme o tamanho do pedaço. A falha não aparecia em teste algum, porque o invariante dos dois sorteios é enunciado **por passe** e continuava valendo. Só é detectável comparando uma execução fatiada com a execução inteira.

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
