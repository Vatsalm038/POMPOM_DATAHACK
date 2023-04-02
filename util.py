import tensorflow as tf
import numpy as np

model = None
output_class = ["Daiatsu_Core", "Daiatsu_Hijet", "Daiatsu_Mira", "FAW_V2", "FAW_XPV", "Honda_BRV", "Honda_City_aspire", "Honda_Grace", "Honda_Vezell", "Honda_city_1994", "Honda_city_2000", "Honda_civic_1994", "Honda_civic_2005", "Honda_civic_2007", "Honda_civic_2015", "Honda_civic_2018", "KIA_Sportage", "Suzuki_Every", "Suzuki_Mehran", "Suzuki_alto_2007", "Suzuki_alto_2019", "Suzuki_alto_japan_2010", "Suzuki_carry", "Suzuki_cultus_2018", "Suzuki_cultus_2019", "Suzuki_highroof", "Suzuki_kyber", "Suzuki_liana", "Suzuki_margala", "Suzuki_swift", "Suzuki_wagonR_2015", "Toyota HIACE 2000", "Toyota_Aqua", "Toyota_Hiace_2012", "Toyota_Landcruser", "Toyota_Passo", "Toyota_Prado", "Toyota_Vigo", "Toyota_Vitz", "Toyota_Vitz_2010", "Toyota_axio", "Toyota_corolla_2000", "Toyota_corolla_2007", "Toyota_corolla_2011", "Toyota_corolla_2016", "Toyota_fortuner", "Toyota_pirus", "Toyota_premio"]
def load_artifacts():
    global model
    model = tf.keras.models.load_model("my_keras_model.h5")

def classify_waste(image_path):
	global model, output_class
	test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
	test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
	test_image = np.expand_dims(test_image, axis = 0)
	predicted_array = model.predict(test_image)
	predicted_value = output_class[np.argmax(predicted_array)]
	return predicted_value