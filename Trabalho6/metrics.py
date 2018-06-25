import numpy as np

def med_Jaccard(filename):
    with open(filename) as f:
        data = f.readlines()

    trackerName = filename.split("/")[-1]
    trackerName = trackerName.split("_")[0]

    Jaccard_indexes = []

    for i in range(len(data)):
        if data[i][0] == 'A' and data[i][2] == 'm':
            Jaccard_indexes.append(float(data[i][23:]))
            #print(float(data[i][23:]))
    
    Jaccard_indexes_np = np.asarray(Jaccard_indexes)
    media_np = np.mean(Jaccard_indexes_np)
    desvio_np = np.std(Jaccard_indexes_np)

    print("A media do Jaccard Index para o tracker {} foi de {:.4f} e o desvio {:.4f}".format(trackerName, media_np, desvio_np))

def med_Robustez(filename):
    with open(filename) as f:
        data = f.readlines()

    trackerName = filename.split("/")[-1]
    trackerName = trackerName.split("_")[0]

    Jaccard_indexes = []

    for i in range(len(data)):
        if data[i][0] == 'A' and data[i][2] == 'r':
            Jaccard_indexes.append(float(data[i][18:]))
            #print(float(data[i][23:]))
    
    Jaccard_indexes_np = np.asarray(Jaccard_indexes)
    media_np = np.mean(Jaccard_indexes_np)
    desvio_np = np.std(Jaccard_indexes_np)

    print("A media da robustez para o tracker {} foi de {:.4f} e o desvio {:.4f}".format(trackerName, media_np, desvio_np))

if __name__ == "__main__":
    filesNames = ["./results/tld_results.txt", "./results/boosting_results.txt", "./results/mil_results.txt",
        "./results/kcf_results.txt", "./results/medianflow_results.txt", "./results/particle_results.txt", "./results/template_results.txt"] 
    
    for filename in filesNames:
        med_Jaccard(filename)
        med_Robustez(filename)