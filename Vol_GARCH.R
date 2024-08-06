
library(rugarch)
library(urca)


#lendo dataframe 
df_vale <- read.csv("Python Scripts/df_vale.csv",
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

# criando coluna com a volatilidade estimada a partir do modelo acima
df_vale$garch_vol <- sigma(fit_garch)

# plot para vizualização dos dados
plot(df_vale$retorno,
     main = "Volatilidade Condicional",
     ylab = "Volatilidade", xlab = "Tempo")

# baixando o arquivo "df_vale" já com a coluna "garch_vol"
write.csv(df_vale_vol, file = "df_vale.csv", row.names = FALSE)
