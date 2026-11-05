class Features : 
    def __init__(self,):
        self.features = {
            "tempo": None,
            "energy": None,
            "density": None,
            "percussion": None,
            "atmosphere": None
        }

    def extract_features(self, heart_rate):
        self.features["tempo"] = sum(heart_rate[-3: -1]) / len(heart_rate[-3: -1])
        self.features["energy"] = 0.65 + (self.features["tempo"] - 90) / 100
        self.features["density"] = 0.7 + (self.features["tempo"] - 90) / 100
        self.features["percussion"] = 0.8 + (self.features["tempo"] - 90) / 100
        self.features["atmosphere"] = 0.4 + (self.features["tempo"] - 90) / 100

