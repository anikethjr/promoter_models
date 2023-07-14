import setuptools

setuptools.setup(
    name='promoter_modelling',
    version='0.1',
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas", "argparse", "wandb", "h5py", "tqdm", "scipy", "scikit-learn", "matplotlib", "seaborn",\
                      "torch==1.13.1", "torchvision", "torchaudio==0.13.1", "torchdata==0.5.1", "torchtext==0.14.1", "torchmtl", "lightning", \
                      "kipoiseq", "pyfaidx", "joblib", "fastsk", "editdistance", "fastdist", "numba", \
                      "transformers", "tensorly", "tensorly-torch", "odfpy", "biopython", "einsum", "rotary-embedding-torch"]
)
