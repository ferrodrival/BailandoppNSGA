# grafs.py
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import Hypervolume
import os
import numpy as np

# Contador interno para las gráficas
_plot_counter = 0

def create_pareto_folder():
    """Crea la carpeta Pareto si no existe"""
    if not os.path.exists("Pareto"):
        os.makedirs("Pareto")

def plot_pareto_front(pareto_front, history=None):
    """Genera y guarda una gráfica 3D del frente de Pareto"""
    global _plot_counter
    create_pareto_folder()
    
    # Incrementar el contador
    _plot_counter += 1
    
    # Configurar la figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extraer los valores de cada objetivo
    V = pareto_front[:, 0]  # Valor
    D = pareto_front[:, 1]  # Diversidad
    N = pareto_front[:, 2]  # Novedad
    
    # Crear el gráfico de dispersión 3D
    scatter = ax.scatter(V, D, N, c='r', marker='o', s=50, alpha=0.6)
    
    # Configurar etiquetas y título
    ax.set_xlabel('Valor (V)', fontsize=12, labelpad=10)
    ax.set_ylabel('Diversidad (D)', fontsize=12, labelpad=10)
    ax.set_zlabel('Novedad (N)', fontsize=12, labelpad=10)
    ax.set_title(f'Frente de Pareto - Ejecución {_plot_counter}', fontsize=14, pad=20)
    
    # Ajustar el ángulo de vista para mejor visualización
    ax.view_init(elev=20, azim=45)
    
    # Guardar la figura
    plt.savefig(f"Pareto/Pareto{_plot_counter}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Si hay datos de historia, graficar convergencia también
    if history is not None:
        plot_convergence(history)

def plot_convergence(history):
    """Genera y guarda una gráfica de convergencia del hipervolumen"""
    global _plot_counter
    create_pareto_folder()
    
    if not history:
        print("Advertencia: No hay datos de historial para graficar convergencia")
        return
    
    # Configurar un punto de referencia fijo adecuado para tu problema
    ref_point = np.array([1.2, 1.2, 1.2])  # Ajusta según tus escalas esperadas
    
    # Calcular hipervolumen para cada generación
    hv = Hypervolume(ref_point=ref_point)
    hypervolumes = []
    
    for algo in history:
        # Obtener el frente Pareto actual (ya convertido a maximización en optimize_dance_sequence)
        F = algo.opt.get("F")
        
        # Verificar y limpiar datos
        if F is None or len(F) == 0:
            continue
            
        # Calcular hipervolumen
        try:
            hv_value = hv.do(F)
            hypervolumes.append(hv_value)
        except:
            print(f"Error calculando HV en generación {len(hypervolumes)}")
            continue
    
    if not hypervolumes:
        print("Error: No se pudo calcular hipervolumen para ninguna generación")
        return
    
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(hypervolumes, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Hipervolumen', fontsize=12)
    plt.title(f'Convergencia del Hipervolumen - Ejecución {_plot_counter}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"Pareto/Convergencia{_plot_counter}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def reset_plot_counter():
    """Reinicia el contador de gráficas (opcional)"""
    global _plot_counter
    _plot_counter = 0