import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def run_inference_on_model(model_path: str, input_image_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    
    interpreter.set_tensor(input_details[0]['index'], [image.transpose(2,0,1).astype('float32')])
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    output_image = output_data[0].transpose(1,2,0)
    
    return output_image
    
    
MODEL_PATH = "../weights/vanilla_unet/sznet_multiclass_v0.tflite"
INPUT_IMAGE = "/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap/images/soriano_15.tif"

input_image = cv2.imread('/Users/conor/Development/AeroSoc/Payload/AI-utils/sznet/datasets/OpenEarthMap/images/soriano_15.tif', cv2.IMREAD_UNCHANGED)
output_image = run_inference_on_model(MODEL_PATH, INPUT_IMAGE)

combined_image = np.zeros((500,500,4))

combined_image[:,:,0:3] = input_image
combined_image[:,:,3] = np.where(output_image[:,:,0] < 1, 0, 1) * 255
# combined_image[:,:,3] = 255

plt.imshow(combined_image.astype('uint8'))
plt.show()