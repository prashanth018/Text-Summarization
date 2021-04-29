{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of text_summword frequencies.ipynb",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/prashanth018/Text-Summarization/blob/shreya/Evaluation_Seq2Seq.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w7nTxB9CVbg9",
        "outputId": "cffff606-2133-4428-f9c8-d87d1a0f07c6"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import nltk\n",
        "import re\n",
        "import bs4 as bs\n",
        "import urllib.request\n",
        "import nltk\n",
        "nltk.download('punkt')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RS9Qk0ynPVzi",
        "outputId": "0f9f4bac-a95f-47d1-89c2-14921264e9b7"
      },
      "source": [
        "!pip install rouge_metric"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: rouge_metric in /usr/local/lib/python3.7/dist-packages (1.0.1)\n",
            "Requirement already satisfied: bert-extractive-summarizer in /usr/local/lib/python3.7/dist-packages (0.7.1)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from bert-extractive-summarizer) (0.22.2.post1)\n",
            "Requirement already satisfied: transformers in /usr/local/lib/python3.7/dist-packages (from bert-extractive-summarizer) (4.5.1)\n",
            "Requirement already satisfied: spacy in /usr/local/lib/python3.7/dist-packages (from bert-extractive-summarizer) (2.2.4)\n",
            "Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->bert-extractive-summarizer) (1.0.1)\n",
            "Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->bert-extractive-summarizer) (1.4.1)\n",
            "Requirement already satisfied: numpy>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->bert-extractive-summarizer) (1.19.5)\n",
            "Requirement already satisfied: tokenizers<0.11,>=0.10.1 in /usr/local/lib/python3.7/dist-packages (from transformers->bert-extractive-summarizer) (0.10.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers->bert-extractive-summarizer) (2019.12.20)\n",
            "Requirement already satisfied: sacremoses in /usr/local/lib/python3.7/dist-packages (from transformers->bert-extractive-summarizer) (0.0.45)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers->bert-extractive-summarizer) (4.41.1)\n",
            "Requirement already satisfied: importlib-metadata; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from transformers->bert-extractive-summarizer) (3.10.1)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers->bert-extractive-summarizer) (3.0.12)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers->bert-extractive-summarizer) (2.23.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from transformers->bert-extractive-summarizer) (20.9)\n",
            "Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (1.0.5)\n",
            "Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (3.0.5)\n",
            "Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (0.4.1)\n",
            "Requirement already satisfied: thinc==7.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (7.4.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (56.0.0)\n",
            "Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (1.0.0)\n",
            "Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (2.0.5)\n",
            "Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (1.0.5)\n",
            "Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (0.8.2)\n",
            "Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.7/dist-packages (from spacy->bert-extractive-summarizer) (1.1.3)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers->bert-extractive-summarizer) (1.15.0)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers->bert-extractive-summarizer) (7.1.2)\n",
            "Requirement already satisfied: typing-extensions>=3.6.4; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from importlib-metadata; python_version < \"3.8\"->transformers->bert-extractive-summarizer) (3.7.4.3)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata; python_version < \"3.8\"->transformers->bert-extractive-summarizer) (3.4.1)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->bert-extractive-summarizer) (2020.12.5)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->bert-extractive-summarizer) (3.0.4)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->bert-extractive-summarizer) (1.24.3)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->bert-extractive-summarizer) (2.10)\n",
            "Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->transformers->bert-extractive-summarizer) (2.4.7)\n",
            "Requirement already satisfied: transformers in /usr/local/lib/python3.7/dist-packages (4.5.1)\n",
            "Requirement already satisfied: tokenizers<0.11,>=0.10.1 in /usr/local/lib/python3.7/dist-packages (from transformers) (0.10.2)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers) (4.41.1)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers) (2.23.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from transformers) (20.9)\n",
            "Requirement already satisfied: importlib-metadata; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from transformers) (3.10.1)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (2019.12.20)\n",
            "Requirement already satisfied: sacremoses in /usr/local/lib/python3.7/dist-packages (from transformers) (0.0.45)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (1.19.5)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers) (3.0.12)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2020.12.5)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2.10)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (1.24.3)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (3.0.4)\n",
            "Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->transformers) (2.4.7)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata; python_version < \"3.8\"->transformers) (3.4.1)\n",
            "Requirement already satisfied: typing-extensions>=3.6.4; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from importlib-metadata; python_version < \"3.8\"->transformers) (3.7.4.3)\n",
            "Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (1.0.1)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (1.15.0)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (7.1.2)\n",
            "Requirement already satisfied: neuralcoref in /usr/local/lib/python3.7/dist-packages (4.0)\n",
            "Requirement already satisfied: boto3 in /usr/local/lib/python3.7/dist-packages (from neuralcoref) (1.17.59)\n",
            "Requirement already satisfied: spacy>=2.1.0 in /usr/local/lib/python3.7/dist-packages (from neuralcoref) (2.2.4)\n",
            "Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.7/dist-packages (from neuralcoref) (2.23.0)\n",
            "Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.7/dist-packages (from neuralcoref) (1.19.5)\n",
            "Requirement already satisfied: botocore<1.21.0,>=1.20.59 in /usr/local/lib/python3.7/dist-packages (from boto3->neuralcoref) (1.20.59)\n",
            "Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.7/dist-packages (from boto3->neuralcoref) (0.10.0)\n",
            "Requirement already satisfied: s3transfer<0.5.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from boto3->neuralcoref) (0.4.2)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (56.0.0)\n",
            "Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (0.8.2)\n",
            "Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (4.41.1)\n",
            "Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (1.1.3)\n",
            "Requirement already satisfied: thinc==7.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (7.4.0)\n",
            "Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (2.0.5)\n",
            "Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (1.0.5)\n",
            "Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (1.0.5)\n",
            "Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (0.4.1)\n",
            "Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (3.0.5)\n",
            "Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.1.0->neuralcoref) (1.0.0)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->neuralcoref) (3.0.4)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->neuralcoref) (1.24.3)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->neuralcoref) (2.10)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->neuralcoref) (2020.12.5)\n",
            "Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /usr/local/lib/python3.7/dist-packages (from botocore<1.21.0,>=1.20.59->boto3->neuralcoref) (2.8.1)\n",
            "Requirement already satisfied: importlib-metadata>=0.20; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.1.0->neuralcoref) (3.10.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil<3.0.0,>=2.1->botocore<1.21.0,>=1.20.59->boto3->neuralcoref) (1.15.0)\n",
            "Requirement already satisfied: typing-extensions>=3.6.4; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=0.20; python_version < \"3.8\"->catalogue<1.1.0,>=0.0.7->spacy>=2.1.0->neuralcoref) (3.7.4.3)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=0.20; python_version < \"3.8\"->catalogue<1.1.0,>=0.0.7->spacy>=2.1.0->neuralcoref) (3.4.1)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XHgYIvY9-xwb",
        "outputId": "bffd0051-0b51-4eaa-818d-1c9f20af90e0"
      },
      "source": [
        "import spacy\n",
        "!python -m spacy download en_core_web_md"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: en_core_web_md==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.2.5/en_core_web_md-2.2.5.tar.gz#egg=en_core_web_md==2.2.5 in /usr/local/lib/python3.7/dist-packages (2.2.5)\n",
            "Requirement already satisfied: spacy>=2.2.2 in /usr/local/lib/python3.7/dist-packages (from en_core_web_md==2.2.5) (2.2.4)\n",
            "Requirement already satisfied: thinc==7.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (7.4.0)\n",
            "Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (1.0.5)\n",
            "Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (0.8.2)\n",
            "Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (1.0.5)\n",
            "Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (2.23.0)\n",
            "Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (1.19.5)\n",
            "Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (1.1.3)\n",
            "Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (0.4.1)\n",
            "Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (3.0.5)\n",
            "Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (1.0.0)\n",
            "Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (2.0.5)\n",
            "Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (4.41.1)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from spacy>=2.2.2->en_core_web_md==2.2.5) (56.0.0)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_md==2.2.5) (2.10)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_md==2.2.5) (3.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_md==2.2.5) (2020.12.5)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_md==2.2.5) (1.24.3)\n",
            "Requirement already satisfied: importlib-metadata>=0.20; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_md==2.2.5) (3.10.1)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=0.20; python_version < \"3.8\"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_md==2.2.5) (3.4.1)\n",
            "Requirement already satisfied: typing-extensions>=3.6.4; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=0.20; python_version < \"3.8\"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_md==2.2.5) (3.7.4.3)\n",
            "\u001b[38;5;2m✔ Download and installation successful\u001b[0m\n",
            "You can now load the model via spacy.load('en_core_web_md')\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GFaJdfCPPhlo",
        "outputId": "492b0646-c00c-45e6-d0f9-0888a4b85d66"
      },
      "source": [
        "!pip install fuzzywuzzy"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: fuzzywuzzy in /usr/local/lib/python3.7/dist-packages (0.18.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7Eqg12AEPpPh",
        "outputId": "2a9e74db-bdbf-4918-822e-793f3561870d"
      },
      "source": [
        "!pip install python-Levenshtein"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: python-Levenshtein in /usr/local/lib/python3.7/dist-packages (0.12.2)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from python-Levenshtein) (56.0.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCkgewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwogICAgICBwZXJjZW50LnRleHRDb250ZW50ID0KICAgICAgICAgIGAke01hdGgucm91bmQoKHBvc2l0aW9uIC8gZmlsZURhdGEuYnl0ZUxlbmd0aCkgKiAxMDApfSUgZG9uZWA7CiAgICB9CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 72
        },
        "id": "ZASozIlXVivE",
        "outputId": "ac985260-25ce-4179-eca1-e172930b4148"
      },
      "source": [
        "from google.colab import files\n",
        "uploaded = files.upload()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-8b615e32-fc35-41ea-946e-62713a4f770d\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-8b615e32-fc35-41ea-946e-62713a4f770d\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "Saving output_v1.csv to output_v1 (2).csv\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 638
        },
        "id": "nrEw_y9UWulB",
        "outputId": "ef53fd93-70ea-4e0c-9ca9-a6fb5d0900ea"
      },
      "source": [
        "dataset = pd.read_csv('output_v1.csv', encoding ='cp1252')\n",
        "dataset['Predicted Summary'] = dataset['Predicted Summary'].str.split(n=1).str[1]\n",
        "dataset['Original Summary'] = dataset['Original Summary'].str.split(n=1).str[1]\n",
        "\n",
        "dataset"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Original Summary</th>\n",
              "      <th>Predicted Summary</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>nsui president accused of sexual harassment re...</td>\n",
              "      <td>will be given to power in mp bjp leader</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>wanted father srk to be as suhana dad daughter...</td>\n",
              "      <td>i was not for my life ayushmann on simmba</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>minor girl and her rapist thrashed in chhattis...</td>\n",
              "      <td>man held for raping woman in delhi boy held</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>can wait for them to make the prime minister o...</td>\n",
              "      <td>i am not to be for not good army chief</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>airtel payments bank losses rise over 11 in fy...</td>\n",
              "      <td>india to invest in india largest fund report</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>union minister slapped at maharashtra event end</td>\n",
              "      <td>ex minister accused of rape accused of rape</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>russia soyuz successfully at iss with astronau...</td>\n",
              "      <td>china launches first ever ever of the world</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>christmas not religious affair for me but spec...</td>\n",
              "      <td>i am not to be for not good cji</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>dad gave permission to bring at home salman end</td>\n",
              "      <td>i am the most expensive in the world games srk</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>rishabh pant dropped from squad for australia ...</td>\n",
              "      <td>kohli 1st indian to slam odi ton in australia ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>lord hanuman was tribal commission chief end</td>\n",
              "      <td>hc rejects plea against twitter for violating</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>not on assam nrc can vote if name on voter lis...</td>\n",
              "      <td>sc rejects plea against cbi chief over rafale ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>iran to deploy warships in ocean amid tensions...</td>\n",
              "      <td>india to build us nuclear capable of us</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13</th>\n",
              "      <td>no brand startup raises million from softbank end</td>\n",
              "      <td>apple posts profit of its first ever ever ever</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>14</th>\n",
              "      <td>cr for tea workers since 2011 wb cm end</td>\n",
              "      <td>govt to give 000 crore to farmers</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15</th>\n",
              "      <td>3 cong workers held for pm modi picture in mp end</td>\n",
              "      <td>bjp leader wing leader arrested over attack on...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16</th>\n",
              "      <td>india japan sign key agree to hold 2 2 dialogu...</td>\n",
              "      <td>pm modi inaugurates india longest oil imports</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17</th>\n",
              "      <td>accidental hindus are now revealing their up c...</td>\n",
              "      <td>pm modi is not to power in 2019 polls bjp</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>18</th>\n",
              "      <td>putin will now have russia on cancelled us tal...</td>\n",
              "      <td>us sanctions will be with us in syria us</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                                     Original Summary                                  Predicted Summary\n",
              "0   nsui president accused of sexual harassment re...            will be given to power in mp bjp leader\n",
              "1   wanted father srk to be as suhana dad daughter...          i was not for my life ayushmann on simmba\n",
              "2   minor girl and her rapist thrashed in chhattis...        man held for raping woman in delhi boy held\n",
              "3   can wait for them to make the prime minister o...             i am not to be for not good army chief\n",
              "4   airtel payments bank losses rise over 11 in fy...       india to invest in india largest fund report\n",
              "5     union minister slapped at maharashtra event end        ex minister accused of rape accused of rape\n",
              "6   russia soyuz successfully at iss with astronau...        china launches first ever ever of the world\n",
              "7   christmas not religious affair for me but spec...                    i am not to be for not good cji\n",
              "8     dad gave permission to bring at home salman end     i am the most expensive in the world games srk\n",
              "9   rishabh pant dropped from squad for australia ...  kohli 1st indian to slam odi ton in australia ...\n",
              "10       lord hanuman was tribal commission chief end      hc rejects plea against twitter for violating\n",
              "11  not on assam nrc can vote if name on voter lis...  sc rejects plea against cbi chief over rafale ...\n",
              "12  iran to deploy warships in ocean amid tensions...            india to build us nuclear capable of us\n",
              "13  no brand startup raises million from softbank end     apple posts profit of its first ever ever ever\n",
              "14            cr for tea workers since 2011 wb cm end                  govt to give 000 crore to farmers\n",
              "15  3 cong workers held for pm modi picture in mp end  bjp leader wing leader arrested over attack on...\n",
              "16  india japan sign key agree to hold 2 2 dialogu...      pm modi inaugurates india longest oil imports\n",
              "17  accidental hindus are now revealing their up c...          pm modi is not to power in 2019 polls bjp\n",
              "18  putin will now have russia on cancelled us tal...           us sanctions will be with us in syria us"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uRMu_JUSPGJf",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d9701712-e79e-4bd0-accd-1d07b2f0d071"
      },
      "source": [
        "hypotheses = list()\n",
        "hypotheses.append(dataset['Predicted Summary'])\n",
        "hypotheses\n",
        "#hypotheses.append(result.split())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[0               will be given to power in mp bjp leader\n",
              " 1             i was not for my life ayushmann on simmba\n",
              " 2           man held for raping woman in delhi boy held\n",
              " 3                i am not to be for not good army chief\n",
              " 4          india to invest in india largest fund report\n",
              " 5           ex minister accused of rape accused of rape\n",
              " 6           china launches first ever ever of the world\n",
              " 7                       i am not to be for not good cji\n",
              " 8        i am the most expensive in the world games srk\n",
              " 9     kohli 1st indian to slam odi ton in australia ...\n",
              " 10        hc rejects plea against twitter for violating\n",
              " 11    sc rejects plea against cbi chief over rafale ...\n",
              " 12              india to build us nuclear capable of us\n",
              " 13       apple posts profit of its first ever ever ever\n",
              " 14                    govt to give 000 crore to farmers\n",
              " 15    bjp leader wing leader arrested over attack on...\n",
              " 16        pm modi inaugurates india longest oil imports\n",
              " 17            pm modi is not to power in 2019 polls bjp\n",
              " 18             us sanctions will be with us in syria us\n",
              " Name: Predicted Summary, dtype: object]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Xe0B9JGKPNdE"
      },
      "source": [
        "references = list()\n",
        "references.append(dataset['Original Summary'])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eFzz7ZCuPPxK",
        "outputId": "7ab995aa-8035-45cb-ce5f-94a4e1f007bc"
      },
      "source": [
        "from rouge_metric import PyRouge\n",
        "rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,\n",
        "                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)\n",
        "scores = rouge.evaluate_tokenized(hypotheses, references)\n",
        "print(scores)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "{'rouge-1': {'r': 0.994088669950739, 'p': 0.06523988102935471, 'f': 0.12244402645470544}, 'rouge-2': {'r': 0.8493975903614458, 'p': 0.054767916100213636, 'f': 0.10290093048713739}, 'rouge-4': {'r': 0.16805845511482254, 'p': 0.010448439223830229, 'f': 0.019673733732510537}, 'rouge-l': {'r': 0.994088669950739, 'p': 0.06523988102935471, 'f': 0.12244402645470544}, 'rouge-w-1.2': {'r': 0.511850937996656, 'p': 0.033650400141646074, 'f': 0.06314920852530527}, 'rouge-s4': {'r': 0.9505219206680584, 'p': 0.0590953338957752, 'f': 0.11127268283741676}, 'rouge-su4': {'r': 0.9580020739716557, 'p': 0.05992950741685767, 'f': 0.1128024583324854}}\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "y3aGezKfPdpE",
        "outputId": "329a1d6e-e31d-400a-8fc5-178cee3dc76f"
      },
      "source": [
        "from fuzzywuzzy import fuzz\n",
        "for i in range(0,19):\n",
        "  fuzz_ratio = fuzz.ratio(dataset['Predicted Summary'].iloc[i], dataset['Original Summary'].iloc[i])\n",
        "  print(fuzz_ratio)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "38\n",
            "42\n",
            "46\n",
            "46\n",
            "43\n",
            "47\n",
            "36\n",
            "39\n",
            "41\n",
            "44\n",
            "34\n",
            "42\n",
            "43\n",
            "34\n",
            "36\n",
            "39\n",
            "38\n",
            "33\n",
            "48\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}