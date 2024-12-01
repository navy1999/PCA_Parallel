# PCA_Parallel
Parallelization of PCA


python -m venv pca_env
source pca_env/bin/activate  # On Windows use: pca_env\Scripts\activate

pip install -r requirements.txt

python parallelPCA.py

#Golang

mkdir parallel_pca
cd parallel_pca
go mod init parallel_pca

touch main.go   # On Windows use: type nul > main.go

go get gonum.org/v1/gonum/mat
go get gonum.org/v1/gonum/stat

go run main.go

OR

go build
./parallel_pca    # On Windows use: parallel_pca.exe



