import setuptools

setuptools.setup(
    name='promoter_modelling',
    version='0.1',
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas", "argparse", "wandb", "h5py", "tqdm", \
                      "scipy", "scikit-learn", "matplotlib", "seaborn",\
                      "torch", "torchmtl", "lightning", \
                      "kipoiseq", "pyfaidx", "joblib", \
                      "transformers", "tensorly", "tensorly-torch", "odfpy", \
                      "biopython", "einsum", "rotary-embedding-torch", "enformer-pytorch", \
                      "boda", "cloudml-hypertune", "dmslogo==0.6.2"]
)
