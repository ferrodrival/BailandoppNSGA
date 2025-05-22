import torch
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from scipy.signal import correlate
import torch.nn.functional as F

class DanceConfig:
    def __init__(self):
        # Parámetros de optimización
        self.pop_size = 50
        self.n_gen = 30
        self.sequence_size = 50
        self.music_features_size = 54
        
        # Pesos para las métricas
        self.value_weights = [0.6, 0.3, 0.1]  # Sincronización, Fluidez, Posturas
        self.diversity_weights = [0.4, 0.4, 0.2]  # MMD, Distancia Promedio, Entropía
        self.novelty_weights = [0.7, 0.3]  # Distancia mínima, Códigos únicos
        
        # Umbrales
        self.novelty_threshold = 0.5
        self.min_valid_angle = -150  # Grados
        self.max_valid_angle = 150   # Grados
        
        # Parámetros de movimiento
        self.max_joint_velocity = 2.0  # Radianes/segundo
        self.collision_threshold = 0.1 # Metros

config = DanceConfig()

class NoveltyMemory:
    def __init__(self, max_size=1000):
        self.memory = []
        self.max_size = max_size
    
    def add(self, sequence):
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)
        self.memory.append(sequence)
    
    def calculate_novelty(self, z):
        if not self.memory:
            return 1.0, 1.0  # Máxima novedad si no hay memoria
        
        if isinstance(z, torch.Tensor):
            z = z.cpu().numpy()
        
        # Distancia mínima a ejemplos previos
        min_dist = min([np.linalg.norm(z - m) for m in self.memory])
        
        # Proporción de códigos no vistos
        z_int = z.astype(int).tolist()
        seen_codes = []
        for m in self.memory:
            if isinstance(m, torch.Tensor):
                seen_codes.extend(m.cpu().numpy().astype(int).tolist())
            else:
                seen_codes.extend(m.astype(int).tolist())
        
        seen = sum(1 for code in z_int if code in seen_codes)
        unique_codes = 1 - (seen / len(z_int)) if len(z_int) > 0 else 1.0
        
        return min_dist, unique_codes

novelty_memory = NoveltyMemory()

def calculate_beat_alignment(pose_changes, music_beats):
    """Calcula la sincronización entre cambios de pose y beats musicales"""
    if len(pose_changes) < 2 or len(music_beats) < 2:
        return 0.0
    
    # Normalizar ambas señales
    pose_norm = (pose_changes - np.mean(pose_changes)) / (np.std(pose_changes) + 1e-8)
    beats_norm = (music_beats - np.mean(music_beats)) / (np.std(music_beats) + 1e-8)
    
    # Calcular correlación cruzada
    corr = correlate(pose_norm, beats_norm, mode='valid')
    if len(corr) > 0:
        peak_corr = np.max(corr) / len(pose_norm)
        return float(peak_corr)
    return 0.0

def validate_poses(poses):
    """Valida que las poses sean biomecánicamente plausibles"""
    # Verificar límites de ángulos articulares
    valid_angles = np.all((poses >= config.min_valid_angle) & (poses <= config.max_valid_angle))
    
    # Verificar velocidad articular (cambios entre frames)
    if len(poses) > 1:
        velocities = np.abs(poses[1:] - poses[:-1])
        valid_velocity = np.all(velocities <= config.max_joint_velocity)
    else:
        valid_velocity = True
    
    return float(valid_angles and valid_velocity)

def calculate_value(z, music_features):
    """Calcula V = w1*S + w2*F + w3*P"""
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z).float()
    if isinstance(music_features, np.ndarray):
        music_features = torch.from_numpy(music_features).float()
    
    # Sincronización rítmica (S)
    beats = music_features[:, 53] if music_features.shape[1] > 53 else music_features[:, -1]
    pose_changes = torch.abs(z[1:] - z[:-1]) if len(z) > 1 else torch.zeros(1)
    beat_alignment = calculate_beat_alignment(pose_changes.numpy(), beats.numpy())
    
    # Fluidez del movimiento (F) - mientras menor cambio entre frames, mejor
    if len(z) > 1:
        smoothness = 1.0 / (torch.mean(torch.abs(z[1:] - z[:-1])).item() + 1e-6)
    else:
        smoothness = 1.0
    
    # Validez de posturas (P)
    valid_poses = validate_poses(z.numpy())
    
    # Combinar métricas
    value = (config.value_weights[0] * beat_alignment +
             config.value_weights[1] * smoothness +
             config.value_weights[2] * valid_poses)
    
    return value

def calculate_diversity(population):
    """Calcula D = λ1*MMD(Z) + λ2*E[|Zi-Zj|] + λ3*H(Z)"""
    if len(population) < 2:
        return 0.0
    
    try:
        Z = torch.stack([torch.from_numpy(ind).float() if isinstance(ind, np.ndarray) else ind.float() for ind in population])
    except:
        Z = torch.stack([torch.tensor(ind).float() for ind in population])
    
    # Maximum Mean Discrepancy (simplificado)
    mmd = torch.var(Z).item()
    
    # Distancia promedio entre pares
    avg_dist = torch.mean(torch.cdist(Z, Z)).item()
    
    # Entropía de códigos
    hist = torch.histc(Z.float(), bins=10)
    hist = hist[hist > 0] / (hist.sum() + 1e-8)
    entropy = -torch.sum(hist * torch.log(hist + 1e-8)).item()
    
    diversity = (config.diversity_weights[0] * mmd +
                 config.diversity_weights[1] * avg_dist +
                 config.diversity_weights[2] * entropy)
    
    return diversity

def calculate_novelty(z):
    """Calcula N = γ1*min|z-r| + γ2*(1-|z|/|z∈R|)"""
    min_dist, unique_codes = novelty_memory.calculate_novelty(z)
    novelty = (config.novelty_weights[0] * min_dist +
               config.novelty_weights[1] * unique_codes)
    
    # Añadir a memoria si es suficientemente novedoso
    if novelty > config.novelty_threshold:
        novelty_memory.add(z)
    
    return novelty

class DanceOptimizationProblem(Problem):
    def __init__(self, music_features):
        super().__init__(n_var=config.sequence_size,
                         n_obj=3,
                         n_constr=0,
                         xl=0,
                         xu=500)
        self.music_features = music_features
    
    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        values = np.zeros(pop_size)
        diversities = np.zeros(pop_size)
        novelties = np.zeros(pop_size)
        
        population_list = [X[i] for i in range(pop_size)]
        diversity_score = calculate_diversity(population_list)
        
        for i in range(pop_size):
            z = X[i]
            values[i] = calculate_value(z, self.music_features)
            novelties[i] = calculate_novelty(z)
            diversities[i] = diversity_score
        
        # NSGA3 minimiza, invertimos valor y diversidad (queremos maximizar)
        out["F"] = np.column_stack([-values, -diversities, -novelties])

def optimize_dance_sequence(population, music_features=None):
    """Interfaz principal para optimizar secuencias de baile"""
    if music_features is None:
        music_features = np.random.rand(config.sequence_size, config.music_features_size)
    
    problem = DanceOptimizationProblem(music_features)
    ref_dirs = get_reference_directions("energy", 3, config.pop_size)
    algorithm = NSGA3(pop_size=config.pop_size, ref_dirs=ref_dirs)
    termination = get_termination("n_gen", config.n_gen)
    
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True,
                   save_history=True,
                   X=np.array(population))
    
    # Procesar resultados
    pareto_front = -res.F  # Convertir a maximización
    pareto_pop = res.X

    # Generar gráficas
    from grafs import plot_pareto_front, plot_convergence
    # Generar gráficas (sin necesidad de pasar el número de iteración)
    plot_pareto_front(pareto_front, res.history)
    
    # Análisis de resultados
    analyze_results(pareto_front, pareto_pop)

    # Análisis de resultados
    analyze_results(pareto_front, pareto_pop)
    
    # Seleccionar mejor solución (mayor valor)
    best_idx = np.argmax(pareto_front[:, 0])
    best_solution = pareto_pop[best_idx]
    
    # Formatear salida
    half_len = len(best_solution) // 2
    zs = (
        [torch.from_numpy(best_solution[:half_len]).long().unsqueeze(0).cuda()],
        [torch.from_numpy(best_solution[half_len:]).long().unsqueeze(0).cuda()]
    )
    
    return zs

# Agregada función para análisis de resultados
def analyze_results(pareto_front, pareto_pop):
    print("\n--- Resultados de Optimización ---")
    print(f"Frente Pareto con {len(pareto_front)} soluciones:")
    
    # Estadísticas básicas
    print("\nValor (V):")
    print(f"  Mejor: {np.max(pareto_front[:, 0]):.2f}")
    print(f"  Peor: {np.min(pareto_front[:, 0]):.2f}")
    print(f"  Promedio: {np.mean(pareto_front[:, 0]):.2f}")
    
    print("\nDiversidad (D):")
    print(f"  Mejor: {np.max(pareto_front[:, 1]):.2f}")
    print(f"  Peor: {np.min(pareto_front[:, 1]):.2f}")
    print(f"  Promedio: {np.mean(pareto_front[:, 1]):.2f}")
    
    print("\nNovedad (N):")
    print(f"  Mejor: {np.max(pareto_front[:, 2]):.2f}")
    print(f"  Peor: {np.min(pareto_front[:, 2]):.2f}")
    print(f"  Promedio: {np.mean(pareto_front[:, 2]):.2f}")
    
    # Soluciones en el frente Pareto
    print("\nEjemplos del frente Pareto:")
    for i in range(min(3, len(pareto_front))):
        print(f"Solución {i+1}: V={pareto_front[i, 0]:.2f}, D={pareto_front[i, 1]:.2f}, N={pareto_front[i, 2]:.2f}")

# Función de conveniencia para integración
def main(population, music_features=None):
    return optimize_dance_sequence(population, music_features)