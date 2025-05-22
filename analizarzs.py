# decodificarzs.py
import time
def analizar_elemento(z, indent=0):
    espacio = " " * indent
    if hasattr(z, 'shape'):
        print(f"{espacio}Tensor: shape = {z.shape}, dtype = {z.dtype}")
        print(f"{espacio}Primeros valores:\n{espacio}{z[:5]}")
    elif isinstance(z, (list, tuple)):
        print(f"{espacio}{type(z)} con {len(z)} elementos:")
        for i, subz in enumerate(z):
            print(f"{espacio}  Elemento {i}:")
            analizar_elemento(subz, indent + 4)
    else:
        print(f"{espacio}Elemento tipo desconocido: {type(z)}")
        print(f"{espacio}{z}")

def analizar_zs(zs):
    print("\n--- Análisis de 'zs' ---")
    analizar_elemento(zs)
    print("--- Fin del análisis ---\n")
    #time.sleep(36000)


#PARA LLAMAR:
#from analizarzs import analizar_zs
#analizar_zs(zs)