import py_dss_interface
import numpy as np
import matplotlib.pyplot as plt
dss = py_dss_interface.DSS()

#______________________________________________________________________________#
#                              MODELAGEM - OPENDSS                             #
#______________________________________________________________________________#

# Índice
idx = 0

# Definição das matrizes de multiplicadores de potência ativa (índice 0) e reativa (índice 1) dos elementos
carga_a = [[0.4873,0.4837,0.4167,0.3613,0.3910,0.3842,0.5055,0.5097,0.7013,0.8342,0.7933,0.5522,0.5543,0.5182,0.5194,0.5867,0.5829,0.8999,0.9643,1.0581,1.0000,0.9892,0.8910,0.8344],[0.0868,0.6062,0.5977,0.4528,0.9950,0.9053,1.0968,0.8239,1.2622,1.1966,0.7084,0.2954,0.4950,0.2772,0.2778,0.2091,0.0000,0.3207,0.6883,0.7552,0.7138,0.8834,0.1587,0.2974]]
carga_b = [[0.5253,0.4183,0.3515,0.3325,0.3219,0.3793,0.4764,0.5153,0.6295,0.7729,0.7917,0.5897,0.5543,0.5201,0.5213,0.5880,0.6317,0.9094,0.9404,0.9625,0.9735,0.9349,0.8633,0.8174],[0.2810,0.5243,0.5042,0.4770,1.0044,1.1102,1.1225,1.1180,1.1330,1.1087,1.1357,0.5266,0.5946,0.0000,0.0000,0.0000,0.0000,0.8121,0.5030,1.2063,0.8693,1.0029,0.6162,0.4373]]
carga_c = [[0.5259,0.4593,0.3930,0.3452,0.3542,0.3642,0.4829,0.5000,0.6752,0.8168,0.7667,0.5592,0.5552,0.5389,0.5402,0.6072,0.6117,0.8643,0.9596,0.9794,0.9671,0.9494,0.8302,0.8185],[0.0937,0.0818,0.0700,0.1230,0.3163,0.9268,0.7806,0.8998,0.4819,0.4369,0.8225,0.7008,0.4958,0.0960,0.0962,0.1082,0.3272,0.4623,0.3420,0.0000,0.3447,0.1691,0.0000,0.0000]]

pv = [0, 0, 0, 0, 0.4, 0.7, 0.8, 0.96, 0.97, 0.99, 1, 0.97, 0.95, 0.8, 0.7, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0]
bateria = [[-0.00077,-0.00080,-0.00079,-0.00078,0.29994,0.49993,0.69992,0.89990,0.99991,1.00000,0.99909,0.99990,0.69992,0.49995,0.29997,-0.00002,-0.00001,-2.63374,-2.63374,-2.63374,-0.00081,-0.00083,-0.00084,-0.00084],[0.1728724672,0.1761144219,0.1733969011,0.2222169249,0.640238379,0.5854588796,0.7468891538,0.9595709178,0.9564719905,0.9593325387,1,0.9253873659,0.9105601907,0.7486054827,0.6158283671,0.6493444577,0.3065554231,0.6246483909,0.6462455304,0.6743265793,0.1760667461,0.1794994041,0.2010965435,0.1979976162]]

# Vetores para armazenar resultados
horas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
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

# Simulação

while idx < 24:
  dss.text("clear")

  dss.text('New Circuit.Circuito bus1=bus_001.1.2.3 basekV=0.22')

  # Dados dos cabos e geometria das linhas
  dss.text('New WireData.Fios Rdc=0.52 Rac=0.63 Runits=km Radius=3.338 Radunits=mm')
  dss.text('New LineSpacing.N1 Nconds=3 Nphases=3 Units=cm X=[ -120.00  0.00  60.00] H=[ 1000.00  1000.00  1000.00]')
  dss.text('New LineGeometry.Geometria  Nconds=3  Spacing=N1 Wires=[ Fios, Fios, Fios ] Reduce=n')

  # Linhas
  dss.text('New Line.Line_001_002 bus1=bus_001 bus2=bus_002 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_002_003 bus1=bus_002 bus2=bus_003 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_003_004 bus1=bus_003 bus2=bus_004 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_004_005 bus1=bus_004 bus2=bus_005 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_005_006 bus1=bus_005 bus2=bus_006 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_006_007 bus1=bus_006 bus2=bus_007 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_007_008 bus1=bus_007 bus2=bus_008 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_008_009 bus1=bus_008 bus2=bus_009 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_009_010 bus1=bus_009 bus2=bus_010 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_010_011 bus1=bus_010 bus2=bus_011 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_011_012 bus1=bus_011 bus2=bus_012 Length=25 Units=m Geometry=Geometria phases=3')
  dss.text('New Line.Line_012_013 bus1=bus_012 bus2=bus_013 Length=25 Units=m Geometry=Geometria phases=3')

  # Loads
  dss.text(f'New Load.cargaA bus=bus_013.1 phases = 1 kv=0.22 kw={3.95251*carga_a[0][idx]: .2f} kvar={0.38724*carga_a[1][idx]: .2f} conn=y')
  dss.text(f'New Load.cargaB bus=bus_013.2 phases = 1 kv=0.22 kw={3.95251*carga_b[0][idx]: .2f} kvar={0.38724*carga_b[1][idx]: .2f} conn=y')
  dss.text(f'New Load.cargaC bus=bus_013.3 phases = 1 kv=0.22 kw={3.95251*carga_c[0][idx]: .2f} kvar={0.38724*carga_c[1][idx]: .2f} conn=y')
  dss.text(f'New Load.bateria6 bus=bus_006.1.2.3 phases=3 kv=0.22 kw={1.21518*bateria[0][idx]: .2f} kvar={0.20975*bateria[1][idx]: .2f} conn=y')

  # Gerador PV
  dss.text(f'New Generator.pv6 bus1=bus_006.1.2.3 phases=3 kv=0.22 kw={8*pv[idx]} kvar=0 conn=y')

  # Resolve
  dss.text("Set VoltageBases = [.22]")
  dss.text("CalcVoltageBases")
  dss.text("Set maxiterations = 500")
  dss.text('set mode = snap')
  dss.text("Solve")
  #dss.text("show lineconstant freq=60 units=m")

  vmag = dss.circuit.buses_vmag
  vname = dss.circuit.buses_names

  # Tensões nos barramentos
  tensoes = np.array(vmag).reshape((len(vname),3))

  #Armazena resultados
  carga_kw_l.append(3.95251*(carga_a[0][idx]+carga_b[0][idx]+carga_c[0][idx])/3)
  carga_kvar_l.append(0.38724*(carga_a[1][idx]+carga_b[1][idx]+carga_c[1][idx])/3)
  bateria_kw_l.append(1.21518*bateria[0][idx])
  bateria_kvar_l.append(0.20975*bateria[1][idx])
  pv6_kw_l.append(8*pv[idx])
  circuit_kw_l.append(dss.circuit.total_power[0])
  circuit_kvar_l.append(dss.circuit.total_power[1])


  carga_v_a.append(tensoes[12][0])
  carga_v_b.append(tensoes[12][1])
  carga_v_c.append(tensoes[12][2])
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

fig.show()