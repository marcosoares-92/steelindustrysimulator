"""STEEL INDUSTRY DIGITAL TWIN PACKAGE
Simulate an steel industry energy consume using Deep Learning.

The digital twin is designed to reproduce the behavior from a DAEWOO Steel Co. Ltd
facility in Gwangyang, South Korea, which made its data public.

Marco Cesar Prado Soares, Data Scientist Specialist @ Bayer Crop Science LATAM
marcosoares.feq@gmail.com
marco.soares@bayer.com
"""

from .core import (
    GlobalVars,
    run_simulation,
    visualize_usage_kwh,
    download_excel_with_data
)


def start_simulation(PT = True):
  """This function runs the following sequence of command line interface commands, 
    for copying the GitHub repository containing the simulator, models and packages to a local 
    repository named 'steelindustrysimulator' 
  
    It is equivalent to running the following command in a notebook's cell:
  
    ! git clone https://github.com/marcosoares-92/steelindustrysimulator 'steelindustrysimulator'
    
    The git clone documentation can be found in:
    https://git-scm.com/docs/git-clone

    : param: PT (boolean): if True, the start message is shown in Portuguese (BR).
    If False, it is shown in English.

  """

  from subprocess import Popen, PIPE, TimeoutExpired
  
  START_MSG = """Starting steel industry operation."""
  START_MSG_PT = """Iniciando operação da indústria de aço."""
  
  if (PT):
    START_MSG = START_MSG_PT

  proc = Popen(["git", "clone", "https://github.com/marcosoares-92/steelindustrysimulator", "steelindustrysimulator"], stdout = PIPE, stderr = PIPE)
  
  try:
      output, error = proc.communicate(timeout = 15)
      print (START_MSG)
  except:
      # General exception
      output, error = proc.communicate()
      print(f"Process with output: {output}, error: {error}.\n")


def digitaltwin_start_msg(PT = True):
    """When the Steel Industry Digital Twin is started, the following message is shown."""

    start_msg = """
        
        Welcome to the Steel Industry Digital Twin!

        This simulator applies advanced AI (deep learning) technologies to reproduce the
        operation of a small-scale steel industry.

        The digital twin is designed to reproduce the behavior from a DAEWOO Steel Co. Ltd
        facility in Gwangyang, South Korea, which made its data public. 
        - This factory produces several types of coils, steel plates, and iron plates. 
        - The information on electricity consumption is held in a cloud-based system. 
        - The information on energy consumption of the industry is stored on the website of the 
        Korea Electric Power Corporation (pccs.kepco.go.kr); and the perspectives on daily, monthly, 
        and annual data are calculated and shown.

        All this information was used for creating the algorithms that will reproduce the energy
        consume behavior based on your user inputs.

        YOUR GOAL HERE IS TO MINIMIZE THE ENERGY CONSUMPTION, WHICH IS SHOWN IN kWh.


        ## For that, you will define:

        - The day of starting the plant simulation (which can be today).
        - The total days and hours for running the plant in the defined conditions
            - Default is 1 day and 0 hours, i.e., 24h of operation.
        
        - Plant operation parameters:
            - Lagging Current reactive power, in kVarh; 
            - Leading Current reactive power, in kVarh; 
            - tCO2(CO2), in ppm; 
            - Lagging Current power factor, in %;
            - Load Type: Light Load, Medium Load, Maximum Load.


        ## And will obtain the Response variable:
            - Energy consumption, in kWh

        - The simulator returns also the following information:
            - Leading Current Power factor, in %; 
            - Number of Seconds from midnight (NSM), in seconds (s); 
            - Week status: if the simulated day is 'Weekend' or 'Weekday'; 
            - Day of week: 'Sunday', 'Monday', ..., 'Saturday'. 
            
    """

    start_msg_pt = """

        Bem-vindo ao gêmeo digital (Digital Twin) da indústria siderúrgica!

         Este simulador aplica tecnologias avançadas de IA (deep learning) para reproduzir o
         operação de uma indústria siderúrgica de pequena escala.

         O gêmeo digital foi projetado para reproduzir o comportamento da fábrica da DAEWOO Steel Co.
         em Gwangyang, Coreia do Sul, que tornou seus dados públicos.
         - Esta fábrica produz diversos tipos de bobinas, chapas de aço e chapas de ferro.
         - As informações sobre o consumo de energia elétrica são mantidas em sistema baseado em nuvem.
         - As informações sobre o consumo de energia da indústria estão armazenadas no site da
         Corporação de Energia Elétrica da Coreia (pccs.kepco.go.kr); e as perspectivas diárias, mensais,
         e os dados anuais são calculados e mostrados.

         Todas essas informações foram utilizadas para a criação dos algoritmos que irão reproduzir o
         comportamento de consumo energético com base nas entradas do usuário.

         SEU OBJETIVO AQUI É MINIMIZAR O CONSUMO DE ENERGIA, QUE É MOSTRADO EM kWh.


         ## Para isso, você definirá:

         - O dia de início da simulação da planta (que pode ser hoje).
         - O total de dias e horas para operar a planta nas condições definidas
             - O padrão é 1 dia e 0 horas, ou seja, 24h de operação.
        
         - Parâmetros de operação da planta:
             - Potência reativa de corrente atrasada, em kVarh;
             - Potência reativa de corrente principal, em kVarh;
             - tCO2(CO2), em ppm;
             - Fator de potência da corrente atrasada, em %;
             - Tipo de Carga: Carga Leve ('Light_Load'), Carga Média ('Medium_Load'), Carga Máxima ('Maximum_Load').


         ## E obterá a variável Response:
             - Consumo de energia, em kWh

         - O simulador retorna também as seguintes informações:
             - Fator de potência de corrente principal, em %;
             - Número de Segundos a partir da meia-noite (NSM), em segundos (s);
             - Status da semana: se o dia simulado é Fim de semana (indicado como 'Weekend') ou 
             'Dia de semana' (indicado como 'Weekday'); 
             - Dia da semana: 'Domingo' (indicado como 'Sunday'), 'Segunda-feira' ('Monday'), 
             'Terça-feira' ('Tuesday'), 'Quarta-feira' ('Wednesday'), 'Quinta-feira' ('Thursday'),
             'Sexta-feira' ('Friday'), 'Sábado' ('Saturday').
            
    """
    
    if (PT):
        start_msg = start_msg_pt

    try:
        # only works in Jupyter Notebook:
        from IPython.display import display
        display(start_msg)
            
    except: # regular mode
        print(start_msg)
