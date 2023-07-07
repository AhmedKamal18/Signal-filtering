import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt
import sounddevice as sd    
from scipy.fftpack import fft


N = 3*1024

𝑡 = np.linspace(0, 3, 4*N)

f = np.linspace(0 , 512 , N//2)

                   #  c  ,  d  ,  e  ,  f  ,  g  ,  a  ,  b
third_octave = [130.81, 146.83, 164.81, 174.61, 196, 220, 246.93]
fourth_octave = [261.63, 293.66, 329.63, 349.23, 392, 440, 493.88]

freq_left = [196, 130.81, 146.83, 130.81, 246.93, 220, 220, 196, 130.81, 130.81]
freq_right = [392, 261.63, 293.66, 261.63, 493.88, 440, 440, 392, 261.63, 261.63]
wait_time = [0.01, 0.03, 0.02, 0.03,0.05,0.04,0.01, 0.03, 0.02, 0]
press_time = [0.38,0.15,0.4,0.3,0.5,0.2,0.2, 0.32,0.15,0.4]

notes = 10

x_t = 0

t_i = 0

for i in range(notes):
    T_i = press_time[i]
    func = sin(2*pi*freq_left[i]*t) + sin(2*pi*freq_right[i]*t)
    x = np.where(np.logical_and(t<=t_i + T_i, t>=t_i),func,0)
    x_t += x
    t_i += (wait_time[i] + press_time[i])
    
plt.plot(t,x_t)
plt.title ('Time Domain Signal')
plt.xlabel ('Time')
plt.ylabel ('Amplitude')
plt.show ()
sd.play(x_t, 3*1024)

x_f = fft(x_t)
x_f = 2/N * np.abs(x_f[0:N//2]) #<----

plt.plot(f,x_f)
plt.title ('Frequency Domain Signal')
plt.xlabel ('Frequency')
plt.ylabel ('Amplitude')
plt.show ()
sd.wait()

f_n1 , f_n2 = np.random.randint(1, 512, 2)

n_t = sin(2*pi*f_n1*t) + sin(2*pi*f_n2*t)

x_n = x_t + n_t

plt.plot(t,x_n)
plt.title ('Time Domain Signal With Noise')
plt.xlabel ('Time')
plt.ylabel ('Amplitude')
plt.show ()
sd.play(x_n, 3*1024)


x_n_f = fft(x_n)
x_n_f = 2/N * np.abs(x_n_f[0:N//2])


plt.plot(f,x_n_f)
plt.title ('Frequency Domain Signal With Noise')
plt.xlabel ('Frequency')
plt.ylabel ('Amplitude')
plt.show ()
sd.wait()


max_amp = np.ceil(np.max(x_f))
noise_freqs = np.where(x_n_f>max_amp, 1, 0).nonzero()[0]

# handling same position noise frequency
if noise_freqs.size == 1:
    noise_freqs = np.append(noise_freqs, noise_freqs[0])


noise_freqs = noise_freqs*512 // (N//2)
f1 = noise_freqs[0]
f2 = noise_freqs[1]

filter = sin(2*pi*f1*t) + sin(2*pi*f2*t)
x_filtered = x_n - filter

plt.plot(t,x_filtered)
plt.title ('Time Domain Filtered Signal')
plt.xlabel ('Time')
plt.ylabel ('Amplitude')
plt.show ()
sd.play(x_filtered, 3*1024)

x_filtered_freq = fft(x_filtered)
x_filtered_freq = 2/N * np.abs(x_filtered_freq[0:N//2])

plt.plot(f,x_filtered_freq)
plt.title ('Frequency Domain Filtered Signal')
plt.xlabel ('Frequency')
plt.ylabel ('Amplitude')
plt.show ()
