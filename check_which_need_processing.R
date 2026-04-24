library(readr)
library(tidyverse)

all_pds <- read_csv("./curated_data/VPDB_PDs_withPMID.csv") 

PDs_with_pmid <- all_pds %>% 
  select(`Gene ID`, PMID_clean, `VEuPathDB Project`) %>% 
  distinct() %>% 
  rename(gene_ID = `Gene ID`, pmid = PMID_clean, host_db = `VEuPathDB Project`)

summaries <- read_csv("./out/summaries/extracted/Summary_Quotes_by_model.csv")

genes_with_summary <- summaries %>% 
  select(gene_ID, pmid) %>% distinct() %>%
  mutate(has_summary = TRUE)


pds_from_summaries <- read_csv("./out/summaries/extracted/PD_rows_all_models.csv")


genes_with_pd <- pds_from_summaries  %>% 
  select(gene_ID, pmid) %>% distinct() %>% 
  mutate(has_PD = TRUE)




gene_processing_check <- genes_with_summary %>% 
full_join(genes_with_pd, by = c("pmid", "gene_ID"))

to_process <- gene_processing_check %>% 
  inner_join(PDs_with_pmid, by = c("pmid", "gene_ID")) %>% 
  filter(is.na(has_PD))

write.csv(x = to_process, "./curated_data/outstanding_to_process.csv", row.names = F)
