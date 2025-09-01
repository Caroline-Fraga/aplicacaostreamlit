
import streamlit as st
import pandas as pd
import joblib
import numpy as np
# from pyngrok import ngrok, PyngrokException # Não precisamos do ngrok diretamente no script Streamlit para rodar no Colab com colab-tunnel

# Título da Aplicação
st.title('Previsão de Nota do IMDB para Filmes')

# Carregar o modelo treinado
# Assumindo que o arquivo do modelo 'random_forest_imdb_predictor.pkl' está disponível no ambiente do Colab
try:
    model = joblib.load('random_forest_imdb_predictor.pkl')
    st.success("Modelo de previsão carregado com sucesso!")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    model = None

# Carregar o DataFrame original ou as informações necessárias para pré-processamento
# Em uma aplicação Streamlit standalone, você pode carregar o CSV original
# ou salvar e carregar apenas as informações de mapping para encoding e imputation.
# Para este exemplo no Colab, assumimos que df_model e X_train estão disponíveis globalmente
# após a execução das células anteriores de pré-processamento e modelagem.
# EM UMA APLICAÇÃO STANDALONE, VOCÊ PRECISARIA SALVAR E CARREGAR ESSES OBJETOS!

df_for_encoding = None
all_genres = []
feature_columns = []

if 'df_model' in globals():
    df_for_encoding = df_model.copy() # Usar uma cópia para evitar SettingWithCopyWarning
    # Obter a lista de todos os gêneros únicos do df_model (que foi one-hot encoded)
    original_and_numeric_cols = ['Meta_score', 'No_of_Votes', 'IMDB_Rating', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']
    all_genres = [col for col in df_for_encoding.columns if col not in original_and_numeric_cols]
    st.info("DataFrame para informações de encoding e imputation disponível.")

    # Obter a ordem das colunas das features usadas no treino (de X_train)
    if 'X_train' in globals():
        feature_columns = X_train.columns.tolist()
        st.info("Lista de colunas de features (X_train) disponível.")
    else:
        st.warning("X_train não encontrado. A ordem e inclusão das colunas para previsão podem estar incorretas.")
        # Tentativa de inferir colunas (menos confiável) - remover a coluna target se existir
        feature_columns = [col for col in df_for_encoding.columns if col != 'IMDB_Rating']

else:
    st.error("df_model não encontrado. Não é possível realizar pré-processamento para previsão.")


st.header("Insira as Características do Filme:")

# Adicionar campos de entrada para as características do filme
meta_score = st.number_input('Meta_score', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
no_of_votes = st.number_input('Número de Votos', min_value=0, value=100000, step=1000)
# Para Gênero, podemos usar um selectbox ou um text input com instrução
genre_input = st.text_input('Gênero (separados por vírgula, ex: Action, Adventure)', value='Drama')
director = st.text_input('Diretor', value='Christopher Nolan')
star1 = st.text_input('Estrela Principal (Star1)', value='Christian Bale')
star2 = st.text_input('Estrela (Star2)', value='Heath Ledger')
star3 = st.text_input('Estrela (Star3)', value='Aaron Eckhart')
star4 = st.text_input('Estrela (Star4)', value='Michael Caine')

# Botão para acionar a previsão
if st.button('Prever Nota do IMDB'):
    if model is not None and df_for_encoding is not None and feature_columns:
        try:
            # Preparar os dados de entrada no formato do DataFrame usado no treino
            input_data = {
                'Meta_score': meta_score,
                'No_of_Votes': no_of_votes,
                'Genre': genre_input,
                'Director': director,
                'Star1': star1,
                'Star2': star2,
                'Star3': star3,
                'Star4': star4
            }

            input_df = pd.DataFrame([input_data])

            # --- Aplicar os mesmos passos de pré-processamento ---

            # Conversão numérica e imputação com a média do treino
            input_df['Meta_score'] = pd.to_numeric(input_df['Meta_score'], errors='coerce').fillna(df_for_encoding['Meta_score'].mean())
            # Assumindo No_of_Votes teve NaNs no treino e foi imputado com a média
            input_df['No_of_Votes'] = pd.to_numeric(input_df['No_of_Votes'], errors='coerce').fillna(df_for_encoding['No_of_Votes'].mean())


            # Target Encoding para colunas categóricas
            categorical_cols = ['Director', 'Star1', 'Star2', 'Star3', 'Star4']
            for col in categorical_cols:
                 # Usar a média do df_for_encoding (que representa os dados de treino processados)
                 mean_target_encoding_map = df_for_encoding.groupby(col)['IMDB_Rating'].mean()
                 # Usar a média geral do IMDB_Rating do treino como fallback para novas categorias
                 fallback_mean = df_for_encoding['IMDB_Rating'].mean()
                 input_df[col] = input_df[col].map(mean_target_encoding_map).fillna(fallback_mean)


            # One-Hot Encode Genre
            # Criar um DataFrame com todas as colunas de gênero possíveis (obtidas do df_for_encoding) inicializado com 0
            genre_encoded_input = pd.DataFrame(0, index=input_df.index, columns=all_genres)

            # Preencher com 1 para os gêneros presentes na entrada do usuário
            input_genres_list = [g.strip() for g in genre_input.replace(' ', '').split(',') if g.strip()] # Não filtrar por all_genres aqui para lidar com novos gêneros
            for genre in input_genres_list:
                if genre in genre_encoded_input.columns: # Apenas adicione se o gênero existia nos dados de treino
                    genre_encoded_input[genre] = 1

            # Remover a coluna 'Genre' original
            # input_df = input_df.drop('Genre', axis=1) # Não precisamos remover explicitamente se construirmos o final_input_df corretamente


            # Combinar as features numéricas, target encoded e one-hot encoded
            # Criar o DataFrame final de input com todas as colunas esperadas pelo modelo, na ordem correta
            final_input_df = pd.DataFrame(0, index=input_df.index, columns=feature_columns)

            # Copiar os valores das colunas processadas
            # Colunas numéricas e target encoded (que já foram processadas em input_df)
            cols_to_copy_from_input = ['Meta_score', 'No_of_Votes'] + categorical_cols
            for col in cols_to_copy_from_input:
                 if col in input_df.columns and col in final_input_df.columns:
                      final_input_df[col] = input_df[col]

            # Colunas one-hot encoded de gênero
            for col in all_genres: # Iterar sobre TODOS os gêneros que o modelo espera
                 if col in genre_encoded_input.columns and col in final_input_df.columns:
                      final_input_df[col] = genre_encoded_input[col]
                 elif col in final_input_df.columns:
                      final_input_df[col] = 0 # Gênero não presente na entrada, defina como 0


            # Ensure the order of columns matches the training data (already done by creating final_input_df with feature_columns)


            # Make prediction
            predicted_imdb_rating = model.predict(final_input_df)[0]

            st.subheader(f"Nota Prevista do IMDB: {predicted_imdb_rating:.4f}")

        except Exception as e:
            st.error(f"Ocorreu um erro durante a previsão: {e}")
            st.write("Por favor, verifique os dados de entrada e tente novamente.")

    elif model is None:
        st.error("Modelo de previsão não carregado. Não é possível fazer a previsão.")
    elif df_for_encoding is None:
         st.error("Dados para pré-processamento não disponíveis. Não é possível fazer a previsão.")
    elif not feature_columns:
         st.error("Colunas de features não definidas. Não é possível fazer a previsão.")
