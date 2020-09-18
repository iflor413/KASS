import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="KASS",
    version="1.0.0",
    author="Isaac Flores",
    author_email="floresisaac413@gmail.com",
    description="Keras Advanced Serial Sampler.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iflor413/KASS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)
