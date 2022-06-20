import sys
from cx_Freeze import setup, Executable


setup(
    name="xray_classifier",
    version=0.1,
    description="A Flask App which classifiers whether lung xrays have COVID-19, pneomonia, or are normal. Noah Vendrig 2022",
    executables=[Executable("cli.py",base="Win32GUI")],
)