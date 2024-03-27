from keras.models import load_model
import numpy as np

# Load generator model
generator_model = load_model('cgan_generator_cifar100.keras')

# Số lượng hình ảnh muốn sinh ra
num_images_to_generate = 10

# Số chiều của không gian ẩn (latent space)
latent_dim = 100

# Tạo các điểm ngẫu nhiên trong không gian ẩn
random_latent_vectors = np.random.randn(num_images_to_generate, latent_dim)

# Tạo nhãn ngẫu nhiên cho từng hình ảnh (trong trường hợp CIFAR-100, có 100 nhãn)
random_labels = np.random.randint(0, 10, num_images_to_generate)  # Đảm bảo chỉ số nằm trong phạm vi từ 0 đến 9

# Sinh ra hình ảnh từ các điểm ngẫu nhiên trong không gian ẩn và nhãn tương ứng
generated_images = generator_model.predict([random_latent_vectors, random_labels])

# Hiển thị các hình ảnh sinh ra
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(num_images_to_generate):
    plt.subplot(1, num_images_to_generate, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()
