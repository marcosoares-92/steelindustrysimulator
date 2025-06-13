"""STEEL INDUSTRY DIGITAL TWIN PACKAGE
Simulate an steel industry energy consume using Deep Learning.

The digital twin is designed to reproduce the behavior from a DAEWOO Steel Co. Ltd
facility in Gwangyang, South Korea, which made its data public.

Marco Cesar Prado Soares, Data Scientist Specialist @ Bayer Crop Science LATAM
marcosoares.feq@gmail.com
marco.soares@bayer.com
"""

# Before loading .core models, update Scikit-learn:
"""Install the same Scikit-learn version used to train the Cluster model,
    avoiding compatibility issues
    
    Equivalent to running:
    ! python -m pip install scikit-learn==1.3.1
"""

from subprocess import Popen, PIPE, TimeoutExpired

# Lock the appropriate Python versions for preventing the simulator from stopping.
proc = Popen(["python", "-m", "pip", "install", "scikit-learn==1.3.1"], stdout = PIPE, stderr = PIPE)
proc2 = Popen(["python", "-m", "pip", "install", "tensorflow==2.14.0"], stdout = PIPE, stderr = PIPE)
proc3 = Popen(["python", "-m", "pip", "install", "numpy==1.24.4"], stdout = PIPE, stderr = PIPE)
proc4 = Popen(["python", "-m", "pip", "install", "pandas==2.1.1"], stdout = PIPE, stderr = PIPE)
proc5 = Popen(["python", "-m", "pip", "install", "scipy==1.11.3"], stdout = PIPE, stderr = PIPE)
proc6 = Popen(["python", "-m", "pip", "install", "statsmodels==0.14.0"], stdout = PIPE, stderr = PIPE)
proc7 = Popen(["python", "-m", "pip", "install", "matplotlib==3.8.0"], stdout = PIPE, stderr = PIPE)

start_msg = """  
    ----------------------------------------------------------------------
                      STEEL INDUSTRY DIGITAL TWIN TERMINAL


    Updating Digital Twin system...
        
    ------------------------------------------------------------------------

    """

try:
    output, error = proc.communicate(timeout = 30)
except:
    # General exception
    output, error = proc.communicate()

try:
    output, error = proc2.communicate(timeout = 30)
except:
    # General exception
    output, error = proc2.communicate()
        
print(start_msg)


msg = """
 
    ----------------------------------------------------------------------
                      STEEL INDUSTRY DIGITAL TWIN TERMINAL
        
                                    WARNING!

        System update failed. If the digital twin do not work properly, 
        run a cell declaring:

                        ! pip install sklearn==1.3.1 

    ------------------------------------------------------------------------

    """

# Now, import the modules

from .core import (
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
        ----------------------------------------------------------------------
                          STEEL INDUSTRY DIGITAL TWIN TERMINAL


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
            - Lagging Current reactive power, in kVArh; 
            - Leading Current reactive power, in kVArh; 
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
  
        ------------------------------------------------------------------------

    """

    start_msg_pt = """
        ----------------------------------------------------------------------
                          STEEL INDUSTRY DIGITAL TWIN TERMINAL


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
             - Potência reativa de corrente atrasada, em kVArh;
             - Potência reativa de corrente principal, em kVArh;
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
   
        ------------------------------------------------------------------------

    """
    
    if (PT):
        start_msg = start_msg_pt

    print("\n")
    print(start_msg)


def start_digital_twin(PT = True):
    """Check if the files are in the directory and start the simulation:"""
    try:
        # In case the simulator was installed in the machine and the GitHub
        # is not downloaded, do it:
        start_simulation(PT = PT)
    
    except:
        pass

    digitaltwin_start_msg(PT = PT)
