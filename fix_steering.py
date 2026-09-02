import re

content = open('src/channel_model.py').read()

rician_repl = """    def _compute_steering_vector(
        self,
        azimuth: float,
        elevation: float
    ) -> np.ndarray:
        d = self.element_spacing
        k = 2 * np.pi / self.wavelength

        idx = np.arange(self.num_elements)
        row = idx // self.grid_cols
        col = idx % self.grid_cols
        phase = k * d * (
            col * np.sin(azimuth) * np.cos(elevation) +
            row * np.sin(elevation)
        )
        return np.exp(1j * phase)"""

umi_repl = """    def _compute_steering_vector(
        self, azimuth: float, elevation: float
    ) -> np.ndarray:
        d = self.element_spacing
        k = 2 * np.pi / self.wavelength
        
        idx = np.arange(self.num_elements)
        row = idx // self.grid_cols
        col = idx % self.grid_cols
        phase = k * d * (
            col * np.sin(azimuth) * np.cos(elevation) +
            row * np.sin(elevation)
        )
        return np.exp(1j * phase)"""

# RicianChannel steering vector
content = re.sub(
    r"    def _compute_steering_vector\(\s*self,\s*azimuth: float,\s*elevation: float\s*\) -> np\.ndarray:.*?return a",
    rician_repl,
    content,
    flags=re.DOTALL,
    count=1
)

# ThreeGPPUMiChannel steering vector
content = re.sub(
    r"    def _compute_steering_vector\(\s*self, azimuth: float, elevation: float\s*\) -> np\.ndarray:.*?return a",
    umi_repl,
    content,
    flags=re.DOTALL,
    count=1
)

open('src/channel_model.py', 'w').write(content)
