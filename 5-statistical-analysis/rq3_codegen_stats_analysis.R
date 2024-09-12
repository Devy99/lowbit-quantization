# Clear the environment
rm(list=ls())

# Download the required libraries if not installed
if (!require("exact2x2")) install.packages("exact2x2")
if (!require("xtable")) install.packages("xtable")
if (!require("effsize")) install.packages("effsize")
if (!require("nortest")) install.packages("nortest")

# Load the required libraries
library(exact2x2)
library(xtable)
library(effsize)
library(nortest)

data<-read.csv("./results-rq3/code_generation_results.csv")

# Init results list
res=list(base_precision = c(), quant_precision = c(), lng = c(), model = c(), p.value = c(), OR = c())

languages=c('py', 'java')
models=c('deepseekcoder-1b', 'codellama-7b', 'deepseekcoder-7b', 'codellama-13b', 'deepseekcoder-33b', 'codellama-34b')
quantizations=c('2bit-mixed', '2bit-mixed-finetuned')

for (model in models) {
  for (lng in languages) {
    for (quant in quantizations) {
      baseline_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]=='16bit'),]$pass
      quantization_pass=data[which(data["language"]==lng & data["model"]==model & data["quantization"]==quant),]$pass

      # McNemar test
      mn=mcnemar.exact(quantization_pass,baseline_pass)
      p.value=mn$p.value
      or=mn$estimate

      # Store the results
      res$base_precision=c(res$base_precision,'16bit')
      res$quant_precision=c(res$quant_precision,quant)
      res$lng=c(res$lng,lng)
      res$model=c(res$model,model)
      res$p.value=c(res$p.value,p.value)
      res$OR=c(res$OR,or)
    }

    # Adjust p-values
    res$p.value[which(res$model == model & res$lng == lng)]= p.adjust(res$p.value[which(res$model == model & res$lng == lng)], method = "BH")
  }
}

# Generate the dataframes and export to csv
res=data.frame(res)

# Sort dataframe
res=res[order(match(res$model, c('deepseekcoder-1_3b', 'codellama-7b', 'deepseekcoder-7b', 'codellama-13b', 'deepseekcoder-33b', 'codellama-34b')), res$lng, res$base_precision, match(res$quant_precision, c('16bit', '2bit-mixed', '2bit-mixed-finetuned'))),]
write.csv(res, file = "./results-rq3/stats_analysis.csv")