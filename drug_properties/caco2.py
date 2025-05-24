from admet_ai import ADMETModel

model = ADMETModel()
preds = model.predict(smiles="CC(=O)C(NC(=O)CN)c1ccc(O)cc1")

print(f"Predicted Caco-2 Permeability: {preds['Caco2_Wang']}")