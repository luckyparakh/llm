1. Install Pyenv by following https://realpython.com/intro-to-pyenv/#installing-pyenv
2. Although it will be mentioned once pyenv installs: 
	Add below at .bashrc 
		eval "$(pyenv init -)"
		eval "$(pyenv virtualenv-init -)"

	Add below at .profile
		export PYENV_ROOT="$HOME/.pyenv"
		[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
		eval "$(pyenv init -)"
3.  Install Python 3.11.4 
        pyenv install 3.11.4
4.	Create Project Dir & cd to it
    mkdir <dir_name> && cd <dir_name>
    mkdir pycode && cd pycode
5.	Create vENV
		pyenv virtualenv <python_version> <environment_name>
        pyenv virtualenv 3.11.4 pycode
	Activate vENV
		pyenv activate <environment_name>
		pyenv deactivate (FYI only)
6.	Install Modules
		pip3 install langchain==0.0.352
		pip3 install openai==0.27.8
		pip3 install python-dotenv==1.0.0
		pip3 install tenacity==8.3.0
		pip3 install requests
		pip3 install pyYaml==5.3
		pip3 install ChromaDB
		pip3 install tiktoken
		pip3 install pyboxen
7.	Now run python main.py