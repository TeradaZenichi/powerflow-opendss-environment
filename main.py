import json

class Settings:
    def __init__(self, config):
        self.Δt = config["timestep"]/60
        return


class BESS:
    def __init__(self, config, settings):
        self.settings = settings
        self.Pmax = config["Pmax"]
        self.Emax = config["Emax"]
        self.eff = config["eff"]
        self.soc = config["SoC0"]
        self.E = self.Emax * self.soc 

    def charging(self, Pcharge):
        self.E = min(self.E + self.settings.Δt * Pcharge * self.eff, self.Emax)
    
    def discharging(self, Pdischarge):
        self.E = max(self.E - self.settings.Δt * Pdischarge/self.eff, 0)
    


with open("data/config.json") as f:
    data = json.load(f)

settings = Settings(data["settings"])
bess_list = [BESS(data["BESS"], settings), BESS(data["BESS"], settings)]


a = 1



