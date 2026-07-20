import py_dss_interface
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
dss = py_dss_interface.DSS()

# ______________________________________________________________________________#
#                             LEITURA DOS DADOS                                 #
# ______________________________________________________________________________#

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "data" / "microgrid.json"

with open(file_path) as f:
    data = json.load(f)

# ______________________________________________________________________________#
#                             CLASSE REDE ELETRICA                               #
# ______________________________________________________________________________#
class RedeEletrica:
    def __init__(self, dss, data):
        self.dss = dss
        self.data = data

    def configurar_infraestrutura(self):
        # Configurações iniciais da rede
        self.dss.text("clear")
        self.dss.text('New Circuit.Circuito bus1=bus_001.1.2.3 basekV=0.22')

        # linhas
        self.dss.text(f'New WireData.Fios Rdc={self.data["WIRE"]["Rdc"]} Rac={self.data["WIRE"]["Rac"]} Runits=km Radius={self.data["WIRE"]["Radius"]} Radunits=mm')
        self.dss.text('New LineSpacing.N1 Nconds=3 Nphases=3 Units=cm X=[ -120 0 60] H=[1000 1000 1000]')
        self.dss.text('New LineGeometry.Geometria Nconds=3 Spacing=N1 Wires=[Fios,Fios,Fios] Reduce=n')

    def criar_linhas(self):
        for i in range(1, 12):
            self.dss.text(f'New Line.Line_{i:03d}_{i+1:03d} '
             f'bus1=bus_{i:03d} bus2=bus_{i+1:03d} '
             f'Length=25 Units=m Geometry=Geometria phases=3')
            
    def configurar_elementos(self):
        # Configurações dos elementos da rede
        self.dss.text(f'New Load.cargaA bus={self.data["LOAD1"]["bus"]}.1.2.3 phases=3 kv={self.data["LOAD1"]["kv"]} kw=0 kvar=0 conn=y')
        self.dss.text(f'New Load.bateria6 bus={self.data["BESS1"]["bus"]}.1.2.3 phases=3 kv={self.data["BESS1"]["kv"]} kw=0 kvar=0 conn=y')
        self.dss.text(f'New Generator.pv6 bus1={self.data["PV1"]["bus"]}.1.2.3 phases=3 kv={self.data["PV1"]["kv"]} kw=0 kvar=0 conn=y')

    def resolver_fluxo(self):
        # Configurações de simulação
        self.dss.text("Set VoltageBases=[.22]")
        self.dss.text("CalcVoltageBases")
        self.dss.text("set mode=snap")
        self.dss.text("Set maxiterations=500")
        self.dss.text("Solve")

#______________________________________________________________________________#
#                              MODELAGEM - OPENDSS                             #
#______________________________________________________________________________#

# Índice
idx = 0

# Definição das matrizes de multiplicadores de potência ativa (índice 0) e reativa (índice 1) dos elementos
carga_a = [data["LOAD1"]["multi_p"], data["LOAD1"]["multi_q"]]
pv = data["PV1"]["multi_p"]
bateria = [data["BESS1"]["multi_p"], data["BESS1"]["multi_q"]]

# Vetores para armazenar resultados
horas = list(range(1, 25))
carga_kw_l = []
carga_kvar_l = []
bateria_kw_l = []
bateria_kvar_l = []
pv6_kw_l = []
circuit_kw_l = []
circuit_kvar_l = []

carga_v_a = []
carga_v_b = []
carga_v_c = []
ders_v_a = []
ders_v_b = []
ders_v_c = []

microrrede = RedeEletrica(dss, data)
microrrede.configurar_infraestrutura()
microrrede.criar_linhas()
microrrede.configurar_elementos()

while idx < 24:
    # atualiza cargas
    dss.text(f'Edit Load.cargaA kw={data["LOAD1"]["kw"]*carga_a[0][idx]:.2f} kvar={data["LOAD1"]["kvar"]*carga_a[1][idx]:.2f}')

    # atualiza pv
    dss.text(f'Edit Generator.pv6 kw={data["PV1"]["kw"]*pv[idx]:.2f} kvar=0')

    # atualiza bateria
    dss.text(f'Edit Load.bateria6 '
             f'kw={data["BESS1"]["kw"]*bateria[0][idx]:.2f} '
             f'kvar={data["BESS1"]["kvar"]*bateria[1][idx]:.2f}')

    microrrede.resolver_fluxo()

    vmag = dss.circuit.buses_vmag
    vname = dss.circuit.buses_names

  # Tensões nos barramentos
    tensoes = np.array(vmag).reshape((len(vname),3))

   # Armazena resultados
    carga_kw_l.append(data["LOAD1"]["kw"]*carga_a[0][idx])
    carga_kvar_l.append(data["LOAD1"]["kvar"]*carga_a[1][idx])
    bateria_kw_l.append(data["BESS1"]["kw"]*bateria[0][idx])
    bateria_kvar_l.append(data["BESS1"]["kvar"]*bateria[1][idx])
    pv6_kw_l.append(data["PV1"]["kw"]*pv[idx])
    circuit_kw_l.append(dss.circuit.total_power[0])
    circuit_kvar_l.append(dss.circuit.total_power[1])

    carga_v_a.append(tensoes[11][0])
    carga_v_b.append(tensoes[11][1])
    carga_v_c.append(tensoes[11][2])
    ders_v_a.append(tensoes[5][0])
    ders_v_b.append(tensoes[5][1])
    ders_v_c.append(tensoes[5][2])

    idx = idx + 1

# Transforma listas em np array
carga_kw = np.array(carga_kw_l)
carga_kvar = np.array(carga_kvar_l)
bateria_kw = np.array(bateria_kw_l)
bateria_kvar = np.array(bateria_kvar_l)
pv6_kw = -np.array(pv6_kw_l)
circuit_kw = -np.array(circuit_kw_l)
circuit_kvar = -np.array(circuit_kvar_l)

#______________________________________________________________________________#
#                        GRÁFICOS - POTÊNCIA MODELAGEM                         #
#______________________________________________________________________________#

fig, axs = plt.subplots(2, figsize=(8, 8), constrained_layout = True)
#fig = plt.figure(figsize=(10,60))

axs[0].axhline(y=0, color='k', linestyle='-.')
axs[0].grid(color='lightgrey')
axs[0].plot(horas, carga_kw, color='k', marker='o', label='Carga')
axs[0].bar(horas, circuit_kw , color='lightcoral', label='Rede')
axs[0].bar(horas, -pv6_kw , bottom = (circuit_kw/2)+abs(circuit_kw)/2, color='gold', label='Geração Fotovoltaica')
axs[0].bar(horas, bateria_kw , bottom = -pv6_kw, color='dodgerblue', label='Bateria')
axs[0].set_title('Potência ativa dos elementos - Modelagem')
axs[0].set_xlabel('Tempo [h]')
axs[0].set_ylabel('Potência [kW]')
axs[0].set_xticks(np.arange(1, 24, 1))
axs[0].set_xlim([1,24])
axs[0].legend(loc=1)
axs[0].set_axisbelow(True)

axs[1].axhline(y=0, color='k', linestyle='-.')
axs[1].grid(color='lightgrey')
axs[1].plot(horas, carga_kvar, color='k', marker='o' , label='Carga')
axs[1].bar(horas, circuit_kvar , color='lightcoral', label='Rede')
axs[1].bar(horas, bateria_kvar , bottom = circuit_kvar , color='dodgerblue', label='Bateria')
axs[1].set_title('Potência reativa dos elementos - Modelagem')
axs[1].set_xlabel('Tempo [h]')
axs[1].set_ylabel('Potência [kVAr]')
axs[1].set_xticks(np.arange(1, 24, 1))
axs[1].set_xlim([1,24])
axs[1].legend()
axs[1].set_axisbelow(True)

plt.show()