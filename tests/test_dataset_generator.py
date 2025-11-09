from belavkin.dataset_generator import generate_modular_arithmetic_dataset, generate_modular_composition_dataset

def test_dataset_generator():
    X_arith, y_arith = generate_modular_arithmetic_dataset(10, 117)
    assert X_arith.shape == (10, 2)
    assert y_arith.shape == (10,)

    X_comp, y_comp = generate_modular_composition_dataset(10, 117)
    assert X_comp.shape == (10, 3)
    assert y_comp.shape == (10,)
