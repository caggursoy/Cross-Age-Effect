# import packages
import functions

# define main
def main():

# init screen clearing function
def clear():
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        os.system('clear')
    else:
        os.system('cls')

# run main
if __name__ == '__main__':
    clear()
    print('Running main...')
    while True:
        main()
