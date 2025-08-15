{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fddbf9e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2225, 2)\n"
     ]
    },
    {
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
       "      <th>text</th>\n",
       "      <th>category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Ad sales boost Time Warner profit\\n\\nQuarterly...</td>\n",
       "      <td>business</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Dollar gains on Greenspan speech\\n\\nThe dollar...</td>\n",
       "      <td>business</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Yukos unit buyer faces loan claim\\n\\nThe owner...</td>\n",
       "      <td>business</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>High fuel prices hit BA's profits\\n\\nBritish A...</td>\n",
       "      <td>business</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pernod takeover talk lifts Domecq\\n\\nShares in...</td>\n",
       "      <td>business</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                text  category\n",
       "0  Ad sales boost Time Warner profit\\n\\nQuarterly...  business\n",
       "1  Dollar gains on Greenspan speech\\n\\nThe dollar...  business\n",
       "2  Yukos unit buyer faces loan claim\\n\\nThe owner...  business\n",
       "3  High fuel prices hit BA's profits\\n\\nBritish A...  business\n",
       "4  Pernod takeover talk lifts Domecq\\n\\nShares in...  business"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "\n",
    "# Path to your bbc folder\n",
    "data_path = r\"C:/Users/yakou/Downloads/archive/bbc-fulltext (document classification)/bbc\"\n",
    "\n",
    "# Only keep directories (categories)\n",
    "categories = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]\n",
    "\n",
    "data = []\n",
    "for category in categories:\n",
    "    folder_path = os.path.join(data_path, category)\n",
    "    for filename in os.listdir(folder_path):\n",
    "        file_path = os.path.join(folder_path, filename)\n",
    "        with open(file_path, 'r', encoding='latin1') as f:\n",
    "            text = f.read()\n",
    "            data.append({'text': text, 'category': category})\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "print(df.shape)\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "05a6c71a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\yakou\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\yakou\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n",
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\yakou\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    },
    {
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
       "      <th>text</th>\n",
       "      <th>category</th>\n",
       "      <th>clean_text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Ad sales boost Time Warner profit\\n\\nQuarterly...</td>\n",
       "      <td>business</td>\n",
       "      <td>ad sale boost time warner profit quarterly pro...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Dollar gains on Greenspan speech\\n\\nThe dollar...</td>\n",
       "      <td>business</td>\n",
       "      <td>dollar gain greenspan speech dollar hit highes...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Yukos unit buyer faces loan claim\\n\\nThe owner...</td>\n",
       "      <td>business</td>\n",
       "      <td>yukos unit buyer face loan claim owner embattl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>High fuel prices hit BA's profits\\n\\nBritish A...</td>\n",
       "      <td>business</td>\n",
       "      <td>high fuel price hit ba profit british airway b...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pernod takeover talk lifts Domecq\\n\\nShares in...</td>\n",
       "      <td>business</td>\n",
       "      <td>pernod takeover talk lift domecq share uk drin...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                text  category  \\\n",
       "0  Ad sales boost Time Warner profit\\n\\nQuarterly...  business   \n",
       "1  Dollar gains on Greenspan speech\\n\\nThe dollar...  business   \n",
       "2  Yukos unit buyer faces loan claim\\n\\nThe owner...  business   \n",
       "3  High fuel prices hit BA's profits\\n\\nBritish A...  business   \n",
       "4  Pernod takeover talk lifts Domecq\\n\\nShares in...  business   \n",
       "\n",
       "                                          clean_text  \n",
       "0  ad sale boost time warner profit quarterly pro...  \n",
       "1  dollar gain greenspan speech dollar hit highes...  \n",
       "2  yukos unit buyer face loan claim owner embattl...  \n",
       "3  high fuel price hit ba profit british airway b...  \n",
       "4  pernod takeover talk lift domecq share uk drin...  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import string\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "\n",
    "# Download NLTK resources (run once)\n",
    "nltk.download('stopwords')\n",
    "nltk.download('wordnet')\n",
    "nltk.download('punkt')\n",
    "\n",
    "stop_words = set(stopwords.words('english'))\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "\n",
    "def clean_text(text):\n",
    "    # Lowercase\n",
    "    text = text.lower()\n",
    "    # Remove punctuation\n",
    "    text = text.translate(str.maketrans('', '', string.punctuation))\n",
    "    # Tokenize\n",
    "    tokens = nltk.word_tokenize(text)\n",
    "    # Remove stopwords and lemmatize\n",
    "    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]\n",
    "    return ' '.join(tokens)\n",
    "\n",
    "# Apply cleaning\n",
    "df['clean_text'] = df['text'].apply(clean_text)\n",
    "\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "37bfd86c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9775280898876404\n",
      "\n",
      "Classification Report:\n",
      "                precision    recall  f1-score   support\n",
      "\n",
      "     business       0.97      0.97      0.97       115\n",
      "entertainment       0.99      0.97      0.98        72\n",
      "     politics       0.95      0.97      0.96        76\n",
      "        sport       1.00      0.99      1.00       102\n",
      "         tech       0.98      0.99      0.98        80\n",
      "\n",
      "     accuracy                           0.98       445\n",
      "    macro avg       0.98      0.98      0.98       445\n",
      " weighted avg       0.98      0.98      0.98       445\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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Zem+++7TkiVLdPDgQY0YMUL//e9/tWTJkpt+t/nz58vb21tbt27VpEmTNGbMGLuSpCSNHTtWHTt21O7du1WuXDl16NBB3bt319ChQ2337Ojdu7dt+zVr1ujZZ59Vnz59dPDgQb377ruaN2+exo0bZ7ff0aNH66mnntLevXvVokULPfPMMzp//ryKFSumZcuWSZIOHTqkuLi4bMki8oYCLhY1KVdUqw4mSJJC/DwU4O2u7Scv2LZJz7Rqz29Jqhhi7A8V5F3paWmKOXhAUbXr2o1H1a6jPbt3mRSVY8rvx8qsFqGvr6+mTZumEydOKCUlRceOHdPrr78ud3d3u9hGjRqluLg4Xb16VZs2bVKlSpWMPgTmTXK/cuWKpkyZog0bNigqKkqSVLJkSf3www9699139eKLL0qSxo0bp/r160uShgwZokcffVRXr16Vp6enfHx8VKBAAdukNUnasGGD9u3bp4SEBNukuP/9739avny5li5dattvWlqaFixYoKJF//zXu7u7u7y8vOz25+rqqtGjR9telyhRQps3b9aSJUv01FNP3fD7ValSRSNHjpQkRUREaObMmVq/fr2aNGli2+b555+37WPw4MGKiorS8OHD1axZM0lS37599fzzz9u2HzdunIYMGWK7EqJkyZIaO3asBg0aZPss6VpF7folqePHj9eMGTP0888/65FHHlGRIkUkSYGBgTdta6ampma7z0imi3uuJxrCXHVLFZGPRwGt/v8Eq4j3tR8uiclpdtslJqcpyI//psidxAuJyszMVEBAgN14QMA9Onv2jElROab8fqyMviLPGZmWYB08eFBXr161Szika4lPZGSk7XWVKlVsf74+4z8hIUHFixfPcb87duzQ5cuXs53U1zPZ68LCwuySq5t555139N5779ky4rS0tFtOrv9r3NdjT0hIuOE2169eqFy5st3Y1atXlZSUJD8/P+3YsUPbtm2zq1hlZmbq6tWrSk5OlpeXV7b9ent7y9fXN9tn30p0dLRdYilJ/31thIaNGHVb+4E5WlQM1NbjiTp3xX5+odX6tw0tluxjwC3cjUvc8wqOVf5lWoJ1fUb/119/rXvvvddunYeHhy0ZcnNzs41fPyn/+tDGnPYbEhKijRs3Zlv314qNt7d3ruJcsmSJ+vfvr8mTJysqKkq+vr564403tHXr1pu+769xX4/9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JCTQDgNyQAULyKNatmypzp07Z7vyC/gnli5dqg4dOigzM1ONGzfW2rVrJV17dNV3332nVatWmRwh4FhIsOAUVqxYoebNm8vNzU0rVqy46baPP/74XYrKsc2ePVuvv/66unTpkuNcNY6TvczMTC1fvlwxMTGyWCyqUKGCHn/8cbm6upodmsOIj49XXFycqlatamvV//zzz/Lz81O5cuVMjg5wLCRYcAp/vXlfTnOwrmMO1p84Trl39OhRtWjRQqdPn1bZsmVltVp1+PBhFStWTF9//TXPjwNw20iwAOR7LVq0kNVq1ccff6wiRYpIuvZok2effVYuLi76+uuvTY4QgLMhwUKecOHCBRUqVMjsMOCkvL29tWXLFlWuXNlufM+ePapTp44uX75sUmQAnBVXEcLpTJw4UeHh4WrXrp0kqW3btlq2bJlCQkK0cuVK23PSIK1fv17r16/P8aHYH3zwgUlROR4PDw9dunQp2/jly5ft7sgNALl140kagIN69913bc9AW7dunb755hutXr1azZs316uvvmpydI5j9OjRatq0qdavX6+zZ88qMTHRbsGfHnvsMb344ovaunWrrFarrFartmzZoh49enAxAIB/hBYhnI6np6dtAnLfvn119epVvfvuuzp8+LAefPBBkof/FxISokmTJum5554zOxSHd+HCBXXq1Elffvml7WrL9PR0tWrVSvPmzZO/v7/JEQJwNrQI4XQKFy6sU6dOqVixYlq9erVef/11SdcessqVcX9KS0tT7dq1zQ7DKRQqVEhffPGFjh49qoMHD0q6did8btIK4J+iRQin06ZNG3Xo0EFNmjTRuXPn1Lx5c0nS7t27+YX4Fy+88IIWLlxodhhO4/3331fr1q3Vtm1btW3bVq1bt9Z7771ndlgAnBQVLDidqVOnKjw8XKdOndKkSZPk4+MjSYqLi1PPnj1Njs5xXL16VbNnz9Y333yjKlWqZLvR6JQpU0yKzPEMHz5cU6dO1csvv6yoqChJ0k8//aT+/fvr+PHjtiopAOQWc7CAPKphw4Y3XGexWLRhw4a7GI1ju+eeezRjxgw9/fTTduOLFi3Syy+/rLNnz5oUGQBnRQULTufDDz+86fqOHTvepUgc27fffmt2CE4jMzNTNWvWzDZeo0YNZWRkmBARAGdHBQtOp3Dhwnav09PTlZycLHd3d3l5een8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6Xnb78nKypJ0rU344IMP2q273sq08nQyIF8gwQLgkLy9vVW6dOlcbVu9enXFx8erQIECCg8Pz3Gb8uXLa8uWLerYsaNtbMuWLTfcZ0REhDw9PbV+/Xq98MIL2dZfn3OVmZlpGwsKCtK9996rX3/9Vc8880yO+61QoYIWLFiglJQUWxJ3szgAOCdahACc3sMPP6yoqCi1bt1aa9as0fHjx7V582a99tpr2r59uySpb9+++uCDD/TBBx/o8OHDGjlypA4cOHDDfRYsWFCDBw/WoEGD9OGHH+rYsWPasmWL3n//fUlSYGCgPD09tXr1av3xxx+6ePGipGs3L42Ojtb06dN1+PBh7du3T3PnztWUKVMkSR06dJCLi4u6du2qgwcPauXKlfrf//53h48QgLuNBAuA07NYLFq5cqXq1aunLl26qEyZMmrfvr2OHz+uoKAgSVK7du00YsQIDR48WDVq1NCJEyf00ksv3XS/w4cP1yuvvKIRI0aofPnyateunRISEiRJBQoU0Jtvvql3331XoaGhatWqlSTphRde0Hvvvad58+apcuXKql+/vubNm2e7rYOPj4++/PJLHTx4UJGRkRo2bJgmTpx4B48OADNYrEwIAAAAMBQVLAAAAIORYAEAABiMBAsAAMBgJFgAAAAGI8ECAAAwGAkWAACAwUiwAAAADEaCBQAAYDASLAAAAIORYAEAABiMBAsAAMBgJFgAAAAG+z9rS6w2RWaMVAAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    df['clean_text'], df['category'],\n",
    "    test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "# TF-IDF vectorization\n",
    "vectorizer = TfidfVectorizer(max_features=5000)\n",
    "X_train_tfidf = vectorizer.fit_transform(X_train)\n",
    "X_test_tfidf = vectorizer.transform(X_test)\n",
    "\n",
    "# Logistic Regression\n",
    "clf = LogisticRegression(max_iter=200)\n",
    "clf.fit(X_train_tfidf, y_train)\n",
    "\n",
    "# Predictions\n",
    "y_pred = clf.predict(X_test_tfidf)\n",
    "\n",
    "# Metrics\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(6,5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
    "            xticklabels=clf.classes_,\n",
    "            yticklabels=clf.classes_)\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('True')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4a9915d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found existing installation: transformers 4.44.2\n",
      "Uninstalling transformers-4.44.2:\n",
      "  Successfully uninstalled transformers-4.44.2\n"
     ]
    }
   ],
   "source": [
    "!pip uninstall -y transformers\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a198cbb9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting transformers==4.44.2\n",
      "  Using cached transformers-4.44.2-py3-none-any.whl (9.5 MB)\n",
      "Requirement already satisfied: datasets in c:\\users\\yakou\\anaconda3\\lib\\site-packages (4.0.0)\n",
      "Requirement already satisfied: filelock in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (3.9.0)\n",
      "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (0.34.4)\n",
      "Requirement already satisfied: numpy>=1.17 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (1.26.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (23.0)\n",
      "Requirement already satisfied: pyyaml>=5.1 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (6.0)\n",
      "Requirement already satisfied: regex!=2019.12.17 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (2022.7.9)\n",
      "Requirement already satisfied: requests in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (2.32.4)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (0.6.2)\n",
      "Requirement already satisfied: tokenizers<0.20,>=0.19 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (0.19.1)\n",
      "Requirement already satisfied: tqdm>=4.27 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from transformers==4.44.2) (4.67.1)\n",
      "Requirement already satisfied: pyarrow>=15.0.0 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from datasets) (21.0.0)\n",
      "Requirement already satisfied: dill<0.3.9,>=0.3.0 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from datasets) (0.3.8)\n",
      "Requirement already satisfied: pandas in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from datasets) (1.5.3)\n",
      "Requirement already satisfied: xxhash in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from datasets) (3.5.0)\n",
      "Requirement already satisfied: multiprocess<0.70.17 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from datasets) (0.70.16)\n",
      "Requirement already satisfied: fsspec[http]<=2025.3.0,>=2023.1.0 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from datasets) (2025.3.0)\n",
      "Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (3.8.3)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from huggingface-hub<1.0,>=0.23.2->transformers==4.44.2) (4.14.1)\n",
      "Requirement already satisfied: charset_normalizer<4,>=2 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from requests->transformers==4.44.2) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from requests->transformers==4.44.2) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from requests->transformers==4.44.2) (1.26.16)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from requests->transformers==4.44.2) (2023.5.7)\n",
      "Requirement already satisfied: colorama in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from tqdm>=4.27->transformers==4.44.2) (0.4.6)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from pandas->datasets) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from pandas->datasets) (2022.7)\n",
      "Requirement already satisfied: attrs>=17.3.0 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (22.1.0)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (6.0.2)\n",
      "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (4.0.2)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (1.8.1)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (1.3.3)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (1.2.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\yakou\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)\n",
      "Installing collected packages: transformers\n",
      "Successfully installed transformers-4.44.2\n"
     ]
    }
   ],
   "source": [
    "!pip install transformers==4.44.2 datasets --upgrade\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f9507afa",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yakou\\anaconda3\\Lib\\site-packages\\transformers\\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "752bc26bae584f5586b4e41a192ca702",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/1780 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "796bd191be8f4021bb6868b6e3ff1e39",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/445 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "⚠ No GPU detected. Using small subset for faster training.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n",
      "C:\\Users\\yakou\\anaconda3\\Lib\\site-packages\\transformers\\training_args.py:1525: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead\n",
      "  warnings.warn(\n",
      "C:\\Users\\yakou\\anaconda3\\Lib\\site-packages\\torch\\utils\\data\\dataloader.py:666: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.\n",
      "  warnings.warn(warn_msg)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='39' max='126' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [ 39/126 11:37 < 27:19, 0.05 it/s, Epoch 0.60/2]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Epoch</th>\n",
       "      <th>Training Loss</th>\n",
       "      <th>Validation Loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[6], line 76\u001b[0m\n\u001b[0;32m     65\u001b[0m trainer \u001b[38;5;241m=\u001b[39m Trainer(\n\u001b[0;32m     66\u001b[0m     model\u001b[38;5;241m=\u001b[39mmodel\u001b[38;5;241m.\u001b[39mto(device),\n\u001b[0;32m     67\u001b[0m     args\u001b[38;5;241m=\u001b[39mtraining_args,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     72\u001b[0m     compute_metrics\u001b[38;5;241m=\u001b[39mcompute_metrics\n\u001b[0;32m     73\u001b[0m )\n\u001b[0;32m     75\u001b[0m \u001b[38;5;66;03m# Train\u001b[39;00m\n\u001b[1;32m---> 76\u001b[0m trainer\u001b[38;5;241m.\u001b[39mtrain()\n\u001b[0;32m     78\u001b[0m \u001b[38;5;66;03m# Evaluate\u001b[39;00m\n\u001b[0;32m     79\u001b[0m results \u001b[38;5;241m=\u001b[39m trainer\u001b[38;5;241m.\u001b[39mevaluate()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\trainer.py:1938\u001b[0m, in \u001b[0;36mTrainer.train\u001b[1;34m(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)\u001b[0m\n\u001b[0;32m   1936\u001b[0m         hf_hub_utils\u001b[38;5;241m.\u001b[39menable_progress_bars()\n\u001b[0;32m   1937\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1938\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m inner_training_loop(\n\u001b[0;32m   1939\u001b[0m         args\u001b[38;5;241m=\u001b[39margs,\n\u001b[0;32m   1940\u001b[0m         resume_from_checkpoint\u001b[38;5;241m=\u001b[39mresume_from_checkpoint,\n\u001b[0;32m   1941\u001b[0m         trial\u001b[38;5;241m=\u001b[39mtrial,\n\u001b[0;32m   1942\u001b[0m         ignore_keys_for_eval\u001b[38;5;241m=\u001b[39mignore_keys_for_eval,\n\u001b[0;32m   1943\u001b[0m     )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\trainer.py:2279\u001b[0m, in \u001b[0;36mTrainer._inner_training_loop\u001b[1;34m(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)\u001b[0m\n\u001b[0;32m   2276\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcontrol \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcallback_handler\u001b[38;5;241m.\u001b[39mon_step_begin(args, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mstate, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcontrol)\n\u001b[0;32m   2278\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39maccelerator\u001b[38;5;241m.\u001b[39maccumulate(model):\n\u001b[1;32m-> 2279\u001b[0m     tr_loss_step \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtraining_step(model, inputs)\n\u001b[0;32m   2281\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m (\n\u001b[0;32m   2282\u001b[0m     args\u001b[38;5;241m.\u001b[39mlogging_nan_inf_filter\n\u001b[0;32m   2283\u001b[0m     \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m is_torch_xla_available()\n\u001b[0;32m   2284\u001b[0m     \u001b[38;5;129;01mand\u001b[39;00m (torch\u001b[38;5;241m.\u001b[39misnan(tr_loss_step) \u001b[38;5;129;01mor\u001b[39;00m torch\u001b[38;5;241m.\u001b[39misinf(tr_loss_step))\n\u001b[0;32m   2285\u001b[0m ):\n\u001b[0;32m   2286\u001b[0m     \u001b[38;5;66;03m# if loss is nan or inf simply add the average of previous logged losses\u001b[39;00m\n\u001b[0;32m   2287\u001b[0m     tr_loss \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m tr_loss \u001b[38;5;241m/\u001b[39m (\u001b[38;5;241m1\u001b[39m \u001b[38;5;241m+\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mstate\u001b[38;5;241m.\u001b[39mglobal_step \u001b[38;5;241m-\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_globalstep_last_logged)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\trainer.py:3318\u001b[0m, in \u001b[0;36mTrainer.training_step\u001b[1;34m(self, model, inputs)\u001b[0m\n\u001b[0;32m   3315\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m loss_mb\u001b[38;5;241m.\u001b[39mreduce_mean()\u001b[38;5;241m.\u001b[39mdetach()\u001b[38;5;241m.\u001b[39mto(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39margs\u001b[38;5;241m.\u001b[39mdevice)\n\u001b[0;32m   3317\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcompute_loss_context_manager():\n\u001b[1;32m-> 3318\u001b[0m     loss \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcompute_loss(model, inputs)\n\u001b[0;32m   3320\u001b[0m \u001b[38;5;28;01mdel\u001b[39;00m inputs\n\u001b[0;32m   3321\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m (\n\u001b[0;32m   3322\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39margs\u001b[38;5;241m.\u001b[39mtorch_empty_cache_steps \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   3323\u001b[0m     \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mstate\u001b[38;5;241m.\u001b[39mglobal_step \u001b[38;5;241m%\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39margs\u001b[38;5;241m.\u001b[39mtorch_empty_cache_steps \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m0\u001b[39m\n\u001b[0;32m   3324\u001b[0m ):\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\trainer.py:3363\u001b[0m, in \u001b[0;36mTrainer.compute_loss\u001b[1;34m(self, model, inputs, return_outputs)\u001b[0m\n\u001b[0;32m   3361\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   3362\u001b[0m     labels \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m-> 3363\u001b[0m outputs \u001b[38;5;241m=\u001b[39m model(\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39minputs)\n\u001b[0;32m   3364\u001b[0m \u001b[38;5;66;03m# Save past state if it exists\u001b[39;00m\n\u001b[0;32m   3365\u001b[0m \u001b[38;5;66;03m# TODO: this needs to be fixed and made cleaner later.\u001b[39;00m\n\u001b[0;32m   3366\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39margs\u001b[38;5;241m.\u001b[39mpast_index \u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0\u001b[39m:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1773\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1771\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[0;32m   1772\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1773\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1784\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1779\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[0;32m   1780\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[0;32m   1781\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[0;32m   1782\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[0;32m   1783\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[1;32m-> 1784\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m forward_call(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1786\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   1787\u001b[0m called_always_called_hooks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mset\u001b[39m()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\models\\distilbert\\modeling_distilbert.py:883\u001b[0m, in \u001b[0;36mDistilBertForSequenceClassification.forward\u001b[1;34m(self, input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)\u001b[0m\n\u001b[0;32m    875\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124mr\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    876\u001b[0m \u001b[38;5;124;03mlabels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\u001b[39;00m\n\u001b[0;32m    877\u001b[0m \u001b[38;5;124;03m    Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\u001b[39;00m\n\u001b[0;32m    878\u001b[0m \u001b[38;5;124;03m    config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\u001b[39;00m\n\u001b[0;32m    879\u001b[0m \u001b[38;5;124;03m    `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\u001b[39;00m\n\u001b[0;32m    880\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    881\u001b[0m return_dict \u001b[38;5;241m=\u001b[39m return_dict \u001b[38;5;28;01mif\u001b[39;00m return_dict \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig\u001b[38;5;241m.\u001b[39muse_return_dict\n\u001b[1;32m--> 883\u001b[0m distilbert_output \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdistilbert(\n\u001b[0;32m    884\u001b[0m     input_ids\u001b[38;5;241m=\u001b[39minput_ids,\n\u001b[0;32m    885\u001b[0m     attention_mask\u001b[38;5;241m=\u001b[39mattention_mask,\n\u001b[0;32m    886\u001b[0m     head_mask\u001b[38;5;241m=\u001b[39mhead_mask,\n\u001b[0;32m    887\u001b[0m     inputs_embeds\u001b[38;5;241m=\u001b[39minputs_embeds,\n\u001b[0;32m    888\u001b[0m     output_attentions\u001b[38;5;241m=\u001b[39moutput_attentions,\n\u001b[0;32m    889\u001b[0m     output_hidden_states\u001b[38;5;241m=\u001b[39moutput_hidden_states,\n\u001b[0;32m    890\u001b[0m     return_dict\u001b[38;5;241m=\u001b[39mreturn_dict,\n\u001b[0;32m    891\u001b[0m )\n\u001b[0;32m    892\u001b[0m hidden_state \u001b[38;5;241m=\u001b[39m distilbert_output[\u001b[38;5;241m0\u001b[39m]  \u001b[38;5;66;03m# (bs, seq_len, dim)\u001b[39;00m\n\u001b[0;32m    893\u001b[0m pooled_output \u001b[38;5;241m=\u001b[39m hidden_state[:, \u001b[38;5;241m0\u001b[39m]  \u001b[38;5;66;03m# (bs, dim)\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1773\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1771\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[0;32m   1772\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1773\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1784\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1779\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[0;32m   1780\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[0;32m   1781\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[0;32m   1782\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[0;32m   1783\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[1;32m-> 1784\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m forward_call(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1786\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   1787\u001b[0m called_always_called_hooks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mset\u001b[39m()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\models\\distilbert\\modeling_distilbert.py:703\u001b[0m, in \u001b[0;36mDistilBertModel.forward\u001b[1;34m(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)\u001b[0m\n\u001b[0;32m    700\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m attention_mask \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m    701\u001b[0m         attention_mask \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mones(input_shape, device\u001b[38;5;241m=\u001b[39mdevice)  \u001b[38;5;66;03m# (bs, seq_length)\u001b[39;00m\n\u001b[1;32m--> 703\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtransformer(\n\u001b[0;32m    704\u001b[0m     x\u001b[38;5;241m=\u001b[39membeddings,\n\u001b[0;32m    705\u001b[0m     attn_mask\u001b[38;5;241m=\u001b[39mattention_mask,\n\u001b[0;32m    706\u001b[0m     head_mask\u001b[38;5;241m=\u001b[39mhead_mask,\n\u001b[0;32m    707\u001b[0m     output_attentions\u001b[38;5;241m=\u001b[39moutput_attentions,\n\u001b[0;32m    708\u001b[0m     output_hidden_states\u001b[38;5;241m=\u001b[39moutput_hidden_states,\n\u001b[0;32m    709\u001b[0m     return_dict\u001b[38;5;241m=\u001b[39mreturn_dict,\n\u001b[0;32m    710\u001b[0m )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1773\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1771\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[0;32m   1772\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1773\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1784\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1779\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[0;32m   1780\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[0;32m   1781\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[0;32m   1782\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[0;32m   1783\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[1;32m-> 1784\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m forward_call(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1786\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   1787\u001b[0m called_always_called_hooks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mset\u001b[39m()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\models\\distilbert\\modeling_distilbert.py:464\u001b[0m, in \u001b[0;36mTransformer.forward\u001b[1;34m(self, x, attn_mask, head_mask, output_attentions, output_hidden_states, return_dict)\u001b[0m\n\u001b[0;32m    456\u001b[0m     layer_outputs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_gradient_checkpointing_func(\n\u001b[0;32m    457\u001b[0m         layer_module\u001b[38;5;241m.\u001b[39m\u001b[38;5;21m__call__\u001b[39m,\n\u001b[0;32m    458\u001b[0m         hidden_state,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    461\u001b[0m         output_attentions,\n\u001b[0;32m    462\u001b[0m     )\n\u001b[0;32m    463\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 464\u001b[0m     layer_outputs \u001b[38;5;241m=\u001b[39m layer_module(\n\u001b[0;32m    465\u001b[0m         hidden_state,\n\u001b[0;32m    466\u001b[0m         attn_mask,\n\u001b[0;32m    467\u001b[0m         head_mask[i],\n\u001b[0;32m    468\u001b[0m         output_attentions,\n\u001b[0;32m    469\u001b[0m     )\n\u001b[0;32m    471\u001b[0m hidden_state \u001b[38;5;241m=\u001b[39m layer_outputs[\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]\n\u001b[0;32m    473\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m output_attentions:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1773\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1771\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[0;32m   1772\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1773\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1784\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1779\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[0;32m   1780\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[0;32m   1781\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[0;32m   1782\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[0;32m   1783\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[1;32m-> 1784\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m forward_call(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1786\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   1787\u001b[0m called_always_called_hooks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mset\u001b[39m()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\models\\distilbert\\modeling_distilbert.py:408\u001b[0m, in \u001b[0;36mTransformerBlock.forward\u001b[1;34m(self, x, attn_mask, head_mask, output_attentions)\u001b[0m\n\u001b[0;32m    405\u001b[0m sa_output \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39msa_layer_norm(sa_output \u001b[38;5;241m+\u001b[39m x)  \u001b[38;5;66;03m# (bs, seq_length, dim)\u001b[39;00m\n\u001b[0;32m    407\u001b[0m \u001b[38;5;66;03m# Feed Forward Network\u001b[39;00m\n\u001b[1;32m--> 408\u001b[0m ffn_output \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mffn(sa_output)  \u001b[38;5;66;03m# (bs, seq_length, dim)\u001b[39;00m\n\u001b[0;32m    409\u001b[0m ffn_output: torch\u001b[38;5;241m.\u001b[39mTensor \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moutput_layer_norm(ffn_output \u001b[38;5;241m+\u001b[39m sa_output)  \u001b[38;5;66;03m# (bs, seq_length, dim)\u001b[39;00m\n\u001b[0;32m    411\u001b[0m output \u001b[38;5;241m=\u001b[39m (ffn_output,)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1773\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1771\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[0;32m   1772\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1773\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1784\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1779\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[0;32m   1780\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[0;32m   1781\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[0;32m   1782\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[0;32m   1783\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[1;32m-> 1784\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m forward_call(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1786\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   1787\u001b[0m called_always_called_hooks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mset\u001b[39m()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\models\\distilbert\\modeling_distilbert.py:343\u001b[0m, in \u001b[0;36mFFN.forward\u001b[1;34m(self, input)\u001b[0m\n\u001b[0;32m    342\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mforward\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;28minput\u001b[39m: torch\u001b[38;5;241m.\u001b[39mTensor) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m torch\u001b[38;5;241m.\u001b[39mTensor:\n\u001b[1;32m--> 343\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m apply_chunking_to_forward(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mff_chunk, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mchunk_size_feed_forward, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mseq_len_dim, \u001b[38;5;28minput\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\pytorch_utils.py:239\u001b[0m, in \u001b[0;36mapply_chunking_to_forward\u001b[1;34m(forward_fn, chunk_size, chunk_dim, *input_tensors)\u001b[0m\n\u001b[0;32m    236\u001b[0m     \u001b[38;5;66;03m# concatenate output at same dimension\u001b[39;00m\n\u001b[0;32m    237\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m torch\u001b[38;5;241m.\u001b[39mcat(output_chunks, dim\u001b[38;5;241m=\u001b[39mchunk_dim)\n\u001b[1;32m--> 239\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m forward_fn(\u001b[38;5;241m*\u001b[39minput_tensors)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\models\\distilbert\\modeling_distilbert.py:346\u001b[0m, in \u001b[0;36mFFN.ff_chunk\u001b[1;34m(self, input)\u001b[0m\n\u001b[0;32m    345\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mff_chunk\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;28minput\u001b[39m: torch\u001b[38;5;241m.\u001b[39mTensor) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m torch\u001b[38;5;241m.\u001b[39mTensor:\n\u001b[1;32m--> 346\u001b[0m     x \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mlin1(\u001b[38;5;28minput\u001b[39m)\n\u001b[0;32m    347\u001b[0m     x \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mactivation(x)\n\u001b[0;32m    348\u001b[0m     x \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mlin2(x)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1773\u001b[0m, in \u001b[0;36mModule._wrapped_call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1771\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_compiled_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)  \u001b[38;5;66;03m# type: ignore[misc]\u001b[39;00m\n\u001b[0;32m   1772\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1773\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_call_impl(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\module.py:1784\u001b[0m, in \u001b[0;36mModule._call_impl\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1779\u001b[0m \u001b[38;5;66;03m# If we don't have any hooks, we want to skip the rest of the logic in\u001b[39;00m\n\u001b[0;32m   1780\u001b[0m \u001b[38;5;66;03m# this function, and just call forward.\u001b[39;00m\n\u001b[0;32m   1781\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m (\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_forward_pre_hooks\n\u001b[0;32m   1782\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_backward_pre_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_backward_hooks\n\u001b[0;32m   1783\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m _global_forward_hooks \u001b[38;5;129;01mor\u001b[39;00m _global_forward_pre_hooks):\n\u001b[1;32m-> 1784\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m forward_call(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1786\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   1787\u001b[0m called_always_called_hooks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mset\u001b[39m()\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\nn\\modules\\linear.py:125\u001b[0m, in \u001b[0;36mLinear.forward\u001b[1;34m(self, input)\u001b[0m\n\u001b[0;32m    124\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mforward\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;28minput\u001b[39m: Tensor) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Tensor:\n\u001b[1;32m--> 125\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m F\u001b[38;5;241m.\u001b[39mlinear(\u001b[38;5;28minput\u001b[39m, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mweight, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mbias)\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "##import torch\n",
    "from datasets import Dataset\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Encode labels\n",
    "label_encoder = LabelEncoder()\n",
    "df['label'] = label_encoder.fit_transform(df['category'])\n",
    "\n",
    "# Train/test split\n",
    "train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)\n",
    "\n",
    "# Convert to HF Dataset\n",
    "train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])\n",
    "test_dataset = Dataset.from_pandas(test_df[['clean_text', 'label']])\n",
    "\n",
    "# Tokenizer\n",
    "model_name = \"distilbert-base-uncased\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "\n",
    "def tokenize(batch):\n",
    "    return tokenizer(batch['clean_text'], truncation=True)\n",
    "\n",
    "train_dataset = train_dataset.map(tokenize, batched=True)\n",
    "test_dataset = test_dataset.map(tokenize, batched=True)\n",
    "\n",
    "# If on CPU, use smaller dataset for speed\n",
    "if not torch.cuda.is_available():\n",
    "    print(\"⚠ No GPU detected. Using small subset for faster training.\")\n",
    "    train_dataset = train_dataset.shuffle(seed=42).select(range(min(500, len(train_dataset))))\n",
    "    test_dataset = test_dataset.shuffle(seed=42).select(range(min(100, len(test_dataset))))\n",
    "\n",
    "# Data collator\n",
    "data_collator = DataCollatorWithPadding(tokenizer=tokenizer)\n",
    "\n",
    "# Model\n",
    "num_labels = len(label_encoder.classes_)\n",
    "model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)\n",
    "\n",
    "# Accuracy metric\n",
    "def compute_metrics(eval_pred):\n",
    "    logits, labels = eval_pred\n",
    "    predictions = np.argmax(logits, axis=-1)\n",
    "    return {\"accuracy\": accuracy_score(labels, predictions)}\n",
    "\n",
    "# Training arguments\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "training_args = TrainingArguments(\n",
    "    output_dir=\"./results\",\n",
    "    evaluation_strategy=\"epoch\",\n",
    "    save_strategy=\"no\",\n",
    "    learning_rate=2e-5,\n",
    "    per_device_train_batch_size=8,\n",
    "    per_device_eval_batch_size=8,\n",
    "    num_train_epochs=2,\n",
    "    weight_decay=0.01,\n",
    "    logging_dir='./logs',\n",
    "    logging_steps=50\n",
    ")\n",
    "\n",
    "# Trainer\n",
    "trainer = Trainer(\n",
    "    model=model.to(device),\n",
    "    args=training_args,\n",
    "    train_dataset=train_dataset,\n",
    "    eval_dataset=test_dataset,\n",
    "    tokenizer=tokenizer,\n",
    "    data_collator=data_collator,\n",
    "    compute_metrics=compute_metrics\n",
    ")\n",
    "\n",
    "# Train\n",
    "trainer.train()\n",
    "\n",
    "# Evaluate\n",
    "results = trainer.evaluate()\n",
    "print(\"DistilBERT Accuracy:\", results['eval_accuracy'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "336981ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_small = X_test.iloc[:200]  # just 200 samples\n",
    "y_test_small = y_test_enc[:200]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f0aad0c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████████████████████████████████████████████████████████████████████████████| 25/25 [00:54<00:00,  2.19s/it]\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "from tqdm import tqdm\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "model.to(device)\n",
    "\n",
    "batch_size = 8\n",
    "preds = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i in tqdm(range(0, len(X_test_small), batch_size)):\n",
    "        batch_texts = X_test_small.iloc[i:i+batch_size].tolist()\n",
    "        inputs = tokenizer(batch_texts, truncation=True, padding=True, return_tensors=\"pt\").to(device)\n",
    "        outputs = model(**inputs)\n",
    "        batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()\n",
    "        preds.extend(batch_preds)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "797d3319",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline Accuracy: 0.9775280898876404\n",
      "\n",
      "Classification Report:\n",
      "                precision    recall  f1-score   support\n",
      "\n",
      "     business       0.97      0.97      0.97       115\n",
      "entertainment       0.99      0.97      0.98        72\n",
      "     politics       0.95      0.97      0.96        76\n",
      "        sport       1.00      0.99      1.00       102\n",
      "         tech       0.98      0.99      0.98        80\n",
      "\n",
      "     accuracy                           0.98       445\n",
      "    macro avg       0.98      0.98      0.98       445\n",
      " weighted avg       0.98      0.98      0.98       445\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Baseline Logistic Regression\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "\n",
    "# Accuracy\n",
    "acc_baseline = accuracy_score(y_test, y_pred)\n",
    "print(\"Baseline Accuracy:\", acc_baseline)\n",
    "\n",
    "# Report\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "a923bd4c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📊 Model Performance Comparison\n",
      "Baseline (Logistic Regression) Accuracy: 0.9775\n",
      "DistilBERT Accuracy: 0.3050\n"
     ]
    }
   ],
   "source": [
    "print(\"📊 Model Performance Comparison\")\n",
    "print(f\"Baseline (Logistic Regression) Accuracy: {acc_baseline:.4f}\")\n",
    "print(f\"DistilBERT Accuracy: {acc:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1cf481d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
    "\n",
    "cm = confusion_matrix(y_test_small, preds)\n",
    "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)\n",
    "disp.plot(cmap=\"Blues\", values_format='d')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "4ef4b2a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline model saved!\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "joblib.dump(clf, \"baseline_logreg.pkl\")\n",
    "print(\"Baseline model saved!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "56fc2d42",
   "metadata": {},
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "[enforce fail at inline_container.cc:664] . unexpected pos 13952 vs 13846",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\serialization.py:967\u001b[0m, in \u001b[0;36msave\u001b[1;34m(obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization, _disable_byteorder_record)\u001b[0m\n\u001b[0;32m    966\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m _open_zipfile_writer(f) \u001b[38;5;28;01mas\u001b[39;00m opened_zipfile:\n\u001b[1;32m--> 967\u001b[0m     _save(\n\u001b[0;32m    968\u001b[0m         obj,\n\u001b[0;32m    969\u001b[0m         opened_zipfile,\n\u001b[0;32m    970\u001b[0m         pickle_module,\n\u001b[0;32m    971\u001b[0m         pickle_protocol,\n\u001b[0;32m    972\u001b[0m         _disable_byteorder_record,\n\u001b[0;32m    973\u001b[0m     )\n\u001b[0;32m    974\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\serialization.py:1268\u001b[0m, in \u001b[0;36m_save\u001b[1;34m(obj, zip_file, pickle_module, pickle_protocol, _disable_byteorder_record)\u001b[0m\n\u001b[0;32m   1267\u001b[0m \u001b[38;5;66;03m# Now that it is on the CPU we can directly copy it into the zip file\u001b[39;00m\n\u001b[1;32m-> 1268\u001b[0m zip_file\u001b[38;5;241m.\u001b[39mwrite_record(name, storage, num_bytes)\n",
      "\u001b[1;31mRuntimeError\u001b[0m: [enforce fail at inline_container.cc:863] . PytorchStreamWriter failed writing file data/0: file write failed",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[19], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m model\u001b[38;5;241m.\u001b[39msave_pretrained(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdistilbert_model\u001b[39m\u001b[38;5;124m\"\u001b[39m, safe_serialization\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m)\n\u001b[0;32m      2\u001b[0m tokenizer\u001b[38;5;241m.\u001b[39msave_pretrained(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdistilbert_model\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mDistilBERT model & tokenizer saved!\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\transformers\\modeling_utils.py:2795\u001b[0m, in \u001b[0;36mPreTrainedModel.save_pretrained\u001b[1;34m(self, save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)\u001b[0m\n\u001b[0;32m   2793\u001b[0m         safe_save_file(shard, os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(save_directory, shard_file), metadata\u001b[38;5;241m=\u001b[39m{\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mformat\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpt\u001b[39m\u001b[38;5;124m\"\u001b[39m})\n\u001b[0;32m   2794\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 2795\u001b[0m         save_function(shard, os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(save_directory, shard_file))\n\u001b[0;32m   2797\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m index \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m   2798\u001b[0m     path_to_weights \u001b[38;5;241m=\u001b[39m os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(save_directory, weights_name)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\serialization.py:966\u001b[0m, in \u001b[0;36msave\u001b[1;34m(obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization, _disable_byteorder_record)\u001b[0m\n\u001b[0;32m    963\u001b[0m     f \u001b[38;5;241m=\u001b[39m os\u001b[38;5;241m.\u001b[39mfspath(f)\n\u001b[0;32m    965\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m _use_new_zipfile_serialization:\n\u001b[1;32m--> 966\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m _open_zipfile_writer(f) \u001b[38;5;28;01mas\u001b[39;00m opened_zipfile:\n\u001b[0;32m    967\u001b[0m         _save(\n\u001b[0;32m    968\u001b[0m             obj,\n\u001b[0;32m    969\u001b[0m             opened_zipfile,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    972\u001b[0m             _disable_byteorder_record,\n\u001b[0;32m    973\u001b[0m         )\n\u001b[0;32m    974\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\torch\\serialization.py:798\u001b[0m, in \u001b[0;36m_open_zipfile_writer_file.__exit__\u001b[1;34m(self, *args)\u001b[0m\n\u001b[0;32m    797\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__exit__\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;241m*\u001b[39margs) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 798\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfile_like\u001b[38;5;241m.\u001b[39mwrite_end_of_file()\n\u001b[0;32m    799\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfile_stream \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m    800\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfile_stream\u001b[38;5;241m.\u001b[39mclose()\n",
      "\u001b[1;31mRuntimeError\u001b[0m: [enforce fail at inline_container.cc:664] . unexpected pos 13952 vs 13846"
     ]
    }
   ],
   "source": [
    "model.save_pretrained(\"distilbert_model\", safe_serialization=False)\n",
    "tokenizer.save_pretrained(\"distilbert_model\")\n",
    "print(\"DistilBERT model & tokenizer saved!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "24405ce7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True\n",
      "657\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "print(os.path.exists(\"my_model.pkl\"))  # Should be True\n",
    "print(os.path.getsize(\"my_model.pkl\")) # Should be > 0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "77fe8eaa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['my_model.pkl']"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Example model\n",
    "model = RandomForestClassifier()\n",
    "# ...train your model here...\n",
    "\n",
    "# Save it safely\n",
    "joblib.dump(model, \"my_model.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "33aa5e9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(\"my_model.pkl\", \"rb\") as f:\n",
    "    model = joblib.load(f)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "b96a2570",
   "metadata": {},
   "outputs": [
    {
     "ename": "NotFittedError",
     "evalue": "This RandomForestClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNotFittedError\u001b[0m                            Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[25], line 6\u001b[0m\n\u001b[0;32m      4\u001b[0m model \u001b[38;5;241m=\u001b[39m joblib\u001b[38;5;241m.\u001b[39mload(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmy_model.pkl\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m      5\u001b[0m sample \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mDataFrame([[\u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m2\u001b[39m, \u001b[38;5;241m3\u001b[39m]], columns\u001b[38;5;241m=\u001b[39m[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFeature1\u001b[39m\u001b[38;5;124m\"\u001b[39m,\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFeature2\u001b[39m\u001b[38;5;124m\"\u001b[39m,\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFeature3\u001b[39m\u001b[38;5;124m\"\u001b[39m])\n\u001b[1;32m----> 6\u001b[0m \u001b[38;5;28mprint\u001b[39m(model\u001b[38;5;241m.\u001b[39mpredict(sample))\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\ensemble\\_forest.py:820\u001b[0m, in \u001b[0;36mForestClassifier.predict\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    799\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mpredict\u001b[39m(\u001b[38;5;28mself\u001b[39m, X):\n\u001b[0;32m    800\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    801\u001b[0m \u001b[38;5;124;03m    Predict class for X.\u001b[39;00m\n\u001b[0;32m    802\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    818\u001b[0m \u001b[38;5;124;03m        The predicted classes.\u001b[39;00m\n\u001b[0;32m    819\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 820\u001b[0m     proba \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mpredict_proba(X)\n\u001b[0;32m    822\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_outputs_ \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m    823\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mclasses_\u001b[38;5;241m.\u001b[39mtake(np\u001b[38;5;241m.\u001b[39margmax(proba, axis\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m), axis\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\ensemble\\_forest.py:860\u001b[0m, in \u001b[0;36mForestClassifier.predict_proba\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    838\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mpredict_proba\u001b[39m(\u001b[38;5;28mself\u001b[39m, X):\n\u001b[0;32m    839\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    840\u001b[0m \u001b[38;5;124;03m    Predict class probabilities for X.\u001b[39;00m\n\u001b[0;32m    841\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    858\u001b[0m \u001b[38;5;124;03m        classes corresponds to that in the attribute :term:`classes_`.\u001b[39;00m\n\u001b[0;32m    859\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 860\u001b[0m     check_is_fitted(\u001b[38;5;28mself\u001b[39m)\n\u001b[0;32m    861\u001b[0m     \u001b[38;5;66;03m# Check data\u001b[39;00m\n\u001b[0;32m    862\u001b[0m     X \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_validate_X_predict(X)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:1390\u001b[0m, in \u001b[0;36mcheck_is_fitted\u001b[1;34m(estimator, attributes, msg, all_or_any)\u001b[0m\n\u001b[0;32m   1385\u001b[0m     fitted \u001b[38;5;241m=\u001b[39m [\n\u001b[0;32m   1386\u001b[0m         v \u001b[38;5;28;01mfor\u001b[39;00m v \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mvars\u001b[39m(estimator) \u001b[38;5;28;01mif\u001b[39;00m v\u001b[38;5;241m.\u001b[39mendswith(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m_\u001b[39m\u001b[38;5;124m\"\u001b[39m) \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m v\u001b[38;5;241m.\u001b[39mstartswith(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m__\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m   1387\u001b[0m     ]\n\u001b[0;32m   1389\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m fitted:\n\u001b[1;32m-> 1390\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m NotFittedError(msg \u001b[38;5;241m%\u001b[39m {\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mname\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;28mtype\u001b[39m(estimator)\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m})\n",
      "\u001b[1;31mNotFittedError\u001b[0m: This RandomForestClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "model = joblib.load(\"my_model.pkl\")\n",
    "sample = pd.DataFrame([[1, 2, 3]], columns=[\"Feature1\",\"Feature2\",\"Feature3\"])\n",
    "print(model.predict(sample))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "502300bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-15 15:51:20.496 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\yakou\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-08-15 15:51:20.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:20.999 Session state does not function when running a script without `streamlit run`\n",
      "2025-08-15 15:51:21.015 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.015 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.015 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.015 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.018 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.018 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.018 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.018 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.018 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.024 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.024 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.024 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.024 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.024 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.024 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:51:21.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "from sklearn.exceptions import NotFittedError\n",
    "\n",
    "# App title\n",
    "st.title(\"My ML Prediction App\")\n",
    "\n",
    "# Try to load the trained model\n",
    "try:\n",
    "    model = joblib.load(\"my_model.pkl\")\n",
    "except FileNotFoundError:\n",
    "    st.error(\"Model file not found. Please train and save your model first.\")\n",
    "    st.stop()\n",
    "except EOFError:\n",
    "    st.error(\"Model file is corrupted. Please re-save the trained model.\")\n",
    "    st.stop()\n",
    "\n",
    "# Input fields for features\n",
    "st.header(\"Enter the features:\")\n",
    "feature1 = st.number_input(\"Feature 1\")\n",
    "feature2 = st.number_input(\"Feature 2\")\n",
    "feature3 = st.number_input(\"Feature 3\")\n",
    "\n",
    "# Predict button\n",
    "if st.button(\"Predict\"):\n",
    "    data = pd.DataFrame([[feature1, feature2, feature3]],\n",
    "                        columns=[\"Feature1\", \"Feature2\", \"Feature3\"])\n",
    "    try:\n",
    "        prediction = model.predict(data)\n",
    "        st.success(f\"Prediction: {prediction[0]}\")\n",
    "    except NotFittedError:\n",
    "        st.error(\"The model is not trained yet. Please train the model before predicting.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "068cdbea",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.471 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.471 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.473 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.483 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.483 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.483 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.483 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.483 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.487 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.488 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-08-15 15:52:22.488 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "from sklearn.exceptions import NotFittedError\n",
    "\n",
    "st.title(\"My ML Prediction App\")\n",
    "\n",
    "# Load model safely\n",
    "try:\n",
    "    model = joblib.load(\"my_model.pkl\")\n",
    "except FileNotFoundError:\n",
    "    st.error(\"Model file not found.\")\n",
    "    st.stop()\n",
    "except EOFError:\n",
    "    st.error(\"Model file is corrupted.\")\n",
    "    st.stop()\n",
    "\n",
    "st.header(\"Enter features:\")\n",
    "feature1 = st.number_input(\"Feature 1\")\n",
    "feature2 = st.number_input(\"Feature 2\")\n",
    "feature3 = st.number_input(\"Feature 3\")\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    data = pd.DataFrame([[feature1, feature2, feature3]], columns=[\"Feature1\",\"Feature2\",\"Feature3\"])\n",
    "    try:\n",
    "        prediction = model.predict(data)\n",
    "        st.success(f\"Prediction: {prediction[0]}\")\n",
    "    except NotFittedError:\n",
    "        st.error(\"Model is not trained yet.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b610fcc",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
