library(PharmacoGx)
library(tidyr)
library(dplyr)
library(tibble)



for(ps.name in c("NCI60","CTRPv2")){
    print(paste0("Working on ",ps.name))
        
    ps <- readRDS(paste0("../../data/rawdata/PSETS/PSet_",ps.name,".rds"))
    ps <- updateObject(ps)
    treatment.info <- treatmentInfo(ps)%>% select("treatmentid", "cid")
    aucs <- summarizeSensitivityProfiles(ps,sensitivity.measure='aac_recomputed')
    df <- data.frame(aucs)
    df <- rownames_to_column(df,var="treatmentid")
    df <- merge(treatment.info,df, by="treatmentid")


    write.csv(df,paste0("../../data/procdata/",ps.name,".csv"))

}
