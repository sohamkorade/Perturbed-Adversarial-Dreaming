import numpy as np
import librosa

def load_audio(path, sr=16000):
	audio, _ = librosa.load(path, sr=sr)
	return audio

def save_audio(path, audio, sr=16000):
	import soundfile as sf
	sf.write(path, audio, sr)

def audio_to_image(audio):
	# image is spectrogram
	# audio is waveform

	spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
	spec = librosa.power_to_db(spec, ref=np.max)
	spec = spec.astype(np.float32)
	spec = spec.reshape(-1)

	# make sure size is 4096
	if spec.size < 4096:
		# pad with zeros
		spec = np.pad(spec, (0, 4096 - spec.size), 'constant')
	elif spec.size > 4096:
		# truncate
		spec = spec[:4096]
	# reshape to (4, 32, 32)
	spec = np.reshape(spec, (4, 32, 32))

	return spec

def image_to_audio(image):
	# image is spectrogram
	# audio is waveform

	spec = np.squeeze(image, axis=0)
	spec = librosa.db_to_power(spec)
	audio = librosa.feature.inverse.mel_to_audio(spec, sr=16000, n_fft=2048, hop_length=512, win_length=2048)
	# double the amplitude
	audio = audio * 2
	return audio

if __name__ == '__main__':
	audio = load_audio('test.wav')
	image = audio_to_image(audio)
	print(image.shape)

	# display image
	import matplotlib.pyplot as plt
	plt.imshow(image[0])
	plt.show()

	audio = image_to_audio(image)

	save_audio('test2.wav', audio)