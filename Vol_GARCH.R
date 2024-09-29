
library(rugarch)
library(forecast)
library(urca)
library(ggplot2)
library(xtable)




#lendo dataframe 
df_vale <- read.csv("df_vale.csv",
                    colClasses = c("Date", "integer", "character",
                                   "character", "numeric", "numeric"))

# teste de estacionariedade
adf <- ur.df(df_vale$retorno, selectlags = 'AIC',
      type = c('none', 'drift', 'trend'))

# Série estácionária, rejeitamos a hipótese nula
# de existencia de raiz unitária

summary(adf)

# Ajustando o modelo

garch_spec <- ugarchspec(variance.model = list(model = "sGARCH",
                                               garchOrder = c(1, 1)),
                         mean.model = list(armaOrder = c(0, 0)))

fit_garch <- ugarchfit(spec = garch_spec, data = df_vale$retorno)

fit_garch
# criando coluna com a volatilidade estimada a partir do modelo acima
df_vale$garch_vol <- sigma(fit_garch)

#anualizando a volatilidade
df_vale$garch_vol <- df_vale$garch_vol * sqrt(252)

# Gráfico dos retornos

retorno_vale <- ggplot(df_vale, aes(x = data_pregao, y = retorno)) +
  geom_line(color = "blue") +
  labs(x = "Data",
       y = "Retorno") +
  theme_light()  

#Salvar gráfico
ggsave("grafico_retorno_vale.png", plot = retorno_vale, width = 10, height = 6)


# Volatilidade estimada

vol_vale <- ggplot(df_vale, aes(x = data_pregao, y = garch_vol)) +
  geom_line(color = "blue") +
  labs(x = "Data",
       y = "Volatilidade Condicional") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  theme_light()  

# Salvar o gráfico
ggsave("grafico_volatilidade.png", plot = vol_vale, width = 10, height = 6)


  # Extraindo os residos do modelo
residuals_garch <- residuals(fit_garch)

# Gráfico com os residuos
ggplot(data.frame(residuals_garch), aes(x = residuals_garch)) +
  geom_histogram(binwidth = 0.001, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of GARCH Model Residuals",
       x = "Residuals",
       y = "Frequency") +
  theme_minimal()


# baixando o arquivo "df_vale" já com a coluna "garch_vol"
write.csv(df_vale, file = "df_vale_garch.csv", row.names = FALSE)
  


