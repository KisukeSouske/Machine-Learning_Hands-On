from kai.model import Model

csv_path = "./data_valve_calibration.csv"
model = Model(csv_path, "Vazao_L_min")
model.start_training("Abertura_Valvula_Percentual", learning_rate=0.0003, batch_size=100, epochs=10000, show_plot=True)