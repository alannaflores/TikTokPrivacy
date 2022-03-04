# Developer environment setup
A couple of notes before getting started:
  * The following setup instructions were tested on MacOS.
  * Any line prefixed by `$` is intended to be run in the terminal, but don't include the `$` in the command), lines prefixed with `#` are comments.

Setup instructions:

0. Install git and clone the repo:
  * Follow the instructions [here](https://github.com/git-guides/install-git) to install git on your local machine.
  * Follow the instructions [here](https://docs.github.com/en/get-started/getting-started-with-git) to set up your github credentials.
  * Clone the [repo](https://github.com/alannaflores/TikTokPrivacy).

1. Make sure you have Python 3.8.6 or higher installed.
  * In a terminal run `python3 --version` or `python3.8 --version` if either of these displays `Python 3.8.6` then you are good to go.
  * If you don't have the correct version of Python installed, you can download and install it from [here](https://www.python.org/downloads/macos/) (make sure to select version 3.8.6).
  * After installing Python open a new terminal window and try `python3.8 --version`
  * Make sure to use the above `python3.8` command when you run the following steps.
2. Create a virtual environment and install the required dependencies. Run the following commands in a terminal window:
```
# Setup virtual env
$ python3.8 -m venv /path/to/new/virtual/environment
# Activate virtual env
$ source /path/to/new/virtual/environment/bin/activate
# Install npm (assuming you have homebrew installed)
$ brew install node 
# Install tiktok scraper 
$ npm i -g tiktok-scraper
# Install dependencies
$ pip install -r /path/to/github/requirements.txt
```

3. Grabbing videos from tiktok (make sure that you always run all commands below in a terminal window with the virtual env active):
```
$ cd /path/to/github/TikTokPrivacy
$ tiktok-scraper --help

Usage: tiktok-scraper <command> [options]

Commands:
  tiktok-scraper user [id]     Scrape videos from username. Enter only username
  tiktok-scraper hashtag [id]  Scrape videos from hashtag. Enter hashtag without #
  tiktok-scraper trend         Scrape posts from current trends
  tiktok-scraper music [id]    Scrape posts from a music id number
  tiktok-scraper video [id]    Download single video without the watermark
  tiktok-scraper history       View previous download history
  tiktok-scraper from-file [file] [async]  Scrape users, hashtags, music, videos mentioned
                                in a file. 1 value per 1 line

Examples:
  tiktok-scraper user USERNAME -d -n 100 --session sid_tt=dae32131231
  tiktok-scraper trend -d -n 100 --session sid_tt=dae32131231
  tiktok-scraper hashtag HASHTAG_NAME -d -n 100 --session sid_tt=dae32131231
  tiktok-scraper music MUSIC_ID -d -n 50 --session sid_tt=dae32131231
 
Some that are useful for our project (--t allows us to download meta data to csv):
  tiktok-scraper hashtag adhdtips --t "csv"
  tiktok-scraper hashtag aboutmechallenge  --download images --n 7  --t "csv"

For more information check out their github: https://github.com/drawrowfly/tiktok-scraper
```

4. Run `video2data.py` for the videos you have obtained in order to extract a dataframe of information.

# Developer workflow

1) Create a new branch from `main`:
```
$ cd path/to/github/TikTokPRivacy
# make sure you are on the main branch and have the latest version of main
$ git checkout main
$ git pull origin main
# create your new branch
$ git checkout -b my_cool_feature
```
2) Make your changes and test them out locally (see Testing locally above for how to run the app).
3) Push your changes to github
```
$ git push origin my_cool_feature
```
4) Open a pull request [here](https://github.com/TikTokPrivacy/pulls). Your branch should show up automatically with a button to create a new pull request, if you don't see it then you can manually create a pull request by clicking the `New pull request` button and setting then choosing your branch name (`my_cool_feature`) from the dropdown for the `compare` button (the `base` branch should always be `main`).
5) Add a descriptive title and description to your pull request.
6) Tag the appropriate people to review your pull request.
