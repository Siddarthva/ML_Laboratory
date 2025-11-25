import pandas as pd
import numpy as np

# Load Data
data = pd.read_csv("enjoysport.csv")
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

print(concepts)
print(target)

def candidate_elimination(concepts, target):
    specific = concepts[0].copy()
    general = [["?" for _ in range(len(specific))] for _ in range(len(specific))]

    print("\nInitialization")
    print("Specific:", specific)
    print("General:", general)

    for i, val in enumerate(concepts):
        print(f"\n--- Step {i+1} ---")
        if target[i] == "yes":
            for x in range(len(specific)):
                if val[x] != specific[x]:
                    specific[x] = '?'
                    general[x][x] = '?'
        else:
            for x in range(len(specific)):
                if val[x] != specific[x]:
                    general[x][x] = specific[x]
                else:
                    general[x][x] = '?'

        print("Specific:", specific)
        print("General:", general)

    # Remove unnecessary hypotheses
    general = [g for g in general if g != ['?']*len(specific)]

    return specific, general

s_final, g_final = candidate_elimination(concepts, target)

print("\nFinal Specific Hypothesis:", s_final)
print("\nFinal General Hypothesis:", g_final)
