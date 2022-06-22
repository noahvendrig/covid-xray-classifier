__author__ = 'Noah Vendrig'
__license__ = 'MIT'  # copy of the license available @ https://prodicus.mit-license.org/
__version__ = '4.8'
__email__ = 'noahvendrig@gmail.com'
__github__ = "github.com/noahvendrig"  # @noahvendrig
__course__ = 'Software Design and Development'
__date__ = '21/06/2022'
__description__ = 'Flask App that allows users to classify lung X-Ray images into categories of either COVID-19, Viral Pneomonia or normal.'
__info__ = "info available at: https://github.com/noahvendrig/covid-xray-classifier/readme.md"  # some info available here
__pyver__ = '3.8.10'

from app import application

if __name__ == "__main__":
    print("-----------------------------------------------------------------------------------------------------------------------")
    print(f"Developed by {__author__}, {__date__}")
    print(f"More projects at {__github__}")
    print(f"This project is a {__description__}")
    print("Enjoy :)")
    print("-----------------------------------------------------------------------------------------------------------------------\n")
    print()
    print("Starting application...")
    application()