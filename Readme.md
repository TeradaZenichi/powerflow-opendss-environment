# Powerflow OpenDSS Environment

Modelagem da microrrede laboratorial LabREI em OpenDSS.

Convenções dos dispositivos:

- `p_bess > 0` representa carregamento e `p_bess < 0` descarga;
- `q_bess > 0` e `q_pv > 0` representam injeção na rede;
- o BESS é um `Load` no OpenDSS, portanto apenas o adaptador converte para
  `dss_kvar = -q_bess`;
- limites de potência aparente e perdas reativas são aplicados antes do fluxo
  de potência.
