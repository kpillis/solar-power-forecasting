import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,TimeSeriesSplit,cross_val_score

def train_predict(X_train, y_train,X_test,model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(df,model,step,size,max_train_size,plot_results):
    X = df[:size].fillna(-99).drop("POWER",axis=1).values
    y = df[:size].fillna(-99).POWER
    tscv = TimeSeriesSplit(n_splits=int(size/step),max_train_size = max_train_size)
    y_preds = np.array(1)
    y_tests = np.array(1)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        start_time = time.time()
        algo_time = time.time()-start_time
        y_pred = train_predict(X_train,y_train,X_test,model)
        y_pred = y_pred.clip(min=0)
        y_preds = np.append(y_preds,y_pred)
        y_tests = np.append(y_tests,y_test.values)
    if plot_results:
        plt.figure(figsize=(15, 7))
        plt.plot(y_preds[-2*ONE_WEEK:-1*ONE_WEEK], "g", label="prediction", linewidth=2.0)
        plt.plot(y_tests[-2*ONE_WEEK:-1*ONE_WEEK], label="actual", linewidth=2.0)
        cv = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
        deviation = np.sqrt(cv.std())
        scale=1
        lower = y_preds[-2*ONE_WEEK:-1*ONE_WEEK] - (scale * deviation)
        upper = y_preds[-2*ONE_WEEK:-1*ONE_WEEK] + (scale * deviation)
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        plt.show()
    return (y_preds,y_tests,algo_time)

def evaluate_model_b(df,model,size):
    X = df[:size].dropna(axis=0).drop("POWER",axis=1).values
    y = df[:size].dropna(axis=0).POWER
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    y_pred = train_predict(X_train,y_train,X_test,model)
    return (y_pred,y_test.values)

def do_all(dfs,columns, models,names,rolling_columns,windows,step,size,max_train_size,withRolling=False,plot_results=False):
    scores = []
    for name,model in zip(names,models):
        time_per_model = 0
        y_preds_per_model = np.array(1)
        y_tests_per_model = np.array(1)
        for i in range(len(dfs)):
            df = dfs[i][columns].copy()
            if withRolling:
                for window in windows:
                    for column in rolling_columns:
                        rolling_column = df[column].rolling(window = window)
                        df["ROLLING_MEAN_"+column+"_"+str(window)] = rolling_column.mean().shift(step)
            y_preds,y_tests, algo_time = evaluate_model(df,model,step,size,max_train_size,plot_results)
            #y_preds,y_tests = evaluate_model_b(df,model,size)
            y_preds_per_model = np.append(y_preds_per_model,y_preds)
            y_tests_per_model = np.append(y_tests_per_model,y_tests)
            time_per_model = time_per_model + algo_time
        
        score_per_model = mean_squared_error(y_preds_per_model,y_tests_per_model)
        time_per_model = time_per_model / len(dfs)
        predictions = y_preds_per_model
        expected = y_tests_per_model
        
        forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
        bias = sum(forecast_errors) * 1.0/len(expected)
        nps = np.array([name,np.sqrt(score_per_model),bias,time_per_model/len(dfs)])
        print(nps)
        scores.append(nps)
        
def naive(dfs,max_train_size):
    sum_mse = 0
    sum_time = 0
    for i in range(len(dfs)):
        df = dfs[i][:max_train_size].copy()
        start_time = time.time()
        y_test = df.POWER.values
        y_pred = df.POWER.shift(24).values
        predictions = y_pred[24:]
        expected = y_test[24:]
        forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
        bias = sum(forecast_errors) * 1.0/len(expected)
        print('Bias: %f' % bias)
        sum_mse = sum_mse + mean_squared_error(y_pred[24:],y_test[24:])
        sum_time = sum_time + (time.time()-start_time)
        print("Model:\t{0}\n\tZone:\t{1}\n\tEVA:\t{2}\n\tTime\t{3}\n".format("Naive Model", i+1, np.sqrt(mean_squared_error(y_pred[24:],y_test[24:])),time.time()-start_time))
        print(sum_mse/3)
        print(sum_time/3)
        
def fft(df,max_train_size,plot_size):
    start_time = time.time()

    # Seed the random number generator
    sig = df.POWER[:max_train_size]

    time_step = 1
    period = 24
    time_vec = np.arange(0, plot_size, 1)

    plt.figure(figsize=(20, 3))
    plt.plot(time_vec, sig[:7*24], label='Original signal')

    # The FFT of the signal
    sig_fft = fftpack.fft(sig)

    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)

    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(sig.size, d=time_step)

    # Plot the FFT power
    fig, ax = plt.subplots(1,2,figsize=(20,6))
    ax[0].plot(sample_freq, power)
    ax[0].set_xlabel('Frekvencia [Hz]')
    ax[0].set_ylabel('plower')

    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

    # Check that it does indeed correspond to the frequency that we generate
    # the signal with
    np.allclose(peak_freq, 1./period)
    # An inner plot to show the peak frequency

    ax[0].set_title('Csúcs frekvencia')
    ax[0].plot(freqs[:8], power[:8])

    # scipy.signal.find_peaks_cwt can also be used for more advanced
    # peak detection

    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    filtered_sig = fftpack.ifft(high_freq_fft)

    #plt.figure()
    ax[1].plot(time_vec, sig[:plot_size], label='Eredeti jel')
    ax[1].plot(time_vec, (filtered_sig+abs(sig-filtered_sig).mean())[:plot_size], linewidth=3, label='Szűrt jel')
    ax[1].plot(time_vec, np.sqrt(np.power(sig-filtered_sig,2))[:plot_size], linewidth=3, label='Hiba',color='crimson')
    ax[1].set_xlabel('Idő [s]')
    ax[1].set_ylabel('Amplitúdó')

    plt.legend(loc='best')
    np.sqrt(mean_squared_error(filtered_sig,sig))
    sig2 = df.POWER[max_train_size:max_train_size*2]
    np.sqrt(mean_squared_error(filtered_sig,sig2))
