suppressMessages(suppressWarnings(install.packages("reactablefmtr")))
suppressMessages(suppressWarnings(install.packages("nflfastR")))
suppressMessages(suppressWarnings(install.packages("nflplotR")))

suppressMessages(suppressWarnings(library(tidyverse)))
suppressMessages(suppressWarnings(library(nflfastR)))
suppressMessages(suppressWarnings(library(reactable)))
suppressMessages(suppressWarnings(library(reactablefmtr)))
suppressMessages(suppressWarnings(library(viridis)))
suppressMessages(suppressWarnings(library(scales)))
suppressMessages(suppressWarnings(library(htmlwidgets)))
suppressMessages(suppressWarnings(library(IRdisplay)))

dir.create(file.path("plots/"), showWarnings = FALSE)

## Similarity output

sim_matrix <- read.csv("../input/xt-rankings/percentile_similarity_full_df.csv", row.names=1) 
sim_df <- as.data.frame(sim_matrix) %>% mutate_if(is.numeric, round, digits=3)

sim_df_name <- function(name) {
  first <- stringr::word(name, 1)
  last <- stringr::str_trim(stringr::str_extract(name, " .*"))
  glue::glue("<div style='line-height:31px'><span style ='font-family:Arial;font-weight:bold;font-variant:small-caps;font-size:30px'>{first}</span></div>\n    
<div style='line-height:29px'><span style='font-weight:bold;font-variant:small-caps;font-size:33px'>{last}</div>")
}

base_name_df <- data.frame("name" = rownames(sim_df)) %>%
  left_join(rankings, by = c('name' = 'name')) %>%
    select(c(name, Position, Team)) %>%
  left_join(nflfastR::teams_colors_logos, by = c('Team' = 'team_abbr')) %>%
  select(c(name, Position, Team, team_logo_espn)) %>%
  mutate(name = sim_df_name(name))

  # Function to format the row details
row_details <- function(index) {
  player_name <- rownames(sim_df)[index]
  similarities <- sim_df[index, -ncol(sim_df)]
  top_similar <- sort(similarities, decreasing = TRUE)[2:6]
  least_similar <- sort(similarities, decreasing = FALSE)[1:5]
  
  top_similar_df <- data.frame('Most' = names(top_similar),
                               'Similar' = unname(unlist(top_similar)),
                               'Least'=names(least_similar),
                               'Similar' = unname(unlist(least_similar)))
  
  htmltools::div(
    style = "padding: 16px",
    reactable::reactable(top_similar_df, outlined = TRUE)
  )
}

# Create the reactable table
player_similarity_table <- reactable(
  base_name_df,
  pagination = TRUE,
  highlight = TRUE,
  striped = TRUE,
  #theme = espn(),
  defaultPageSize = 1,
  height=500,
  defaultColDef = colDef(align = "center"),
  searchable = TRUE,
  defaultSortOrder = "desc",
  details = row_details,
  columns = list(
            name = colDef(name = "Name", #maxWidth = 120, 
                          html = TRUE),
            Position = colDef(name = "Position", #maxWidth = 120, 
                         style = list()),
            Team = colDef(name = "Team", maxWidth = 70, 
                         style = list()),
            team_logo_espn = colDef(name = "Logo", maxWidth = 200,
                                    cell = embed_img(height = 150, width = 150))
  )
#   columns = list(
#     Player = colDef(name = "Player", sortable = TRUE, searchable = TRUE)
#   )
)

p0 <-"plots/sim_table.html"
saveWidget(player_similarity_table, file.path(normalizePath(dirname(p0)),basename(p0)))


first_last <- function(name) {
  first <- stringr::word(name, 1)
  last <- stringr::str_trim(stringr::str_extract(name, " .*"))
  glue::glue("<div style='line-height:11px'><span style ='font-family:Arial;font-weight:bold;color:grey;font-size:10px'>{first}</span></div>\n    
<div style='line-height:9px'><span style='font-weight:bold;font-variant:small-caps;font-size:13px'>{last}</div>")
}

rankings <- read.csv("../input/xt-rankings/player_rankings.csv", row.names=1) 

t0 <- rankings %>%
  left_join(nflfastR::teams_colors_logos, by = c('Team' = 'team_abbr')) %>%
  select(c(name, Position, Team, team_logo_espn, Total.Snaps, xT, xT.snap)) %>%
  mutate(xT = round(xT, digits = 3),
         xT.snap = round(xT.snap, digits = 3),
         name = first_last(name))

tackle <- reactable(t0,
          pagination = TRUE,
          highlight = TRUE,
          striped = TRUE,
          defaultSorted = "xT",
          defaultSortOrder = "desc",
          theme = espn(),
          defaultPageSize = 10,
          defaultColDef = colDef(align = "center"),
          columns = list(
            name = colDef(name = "name", maxWidth = 120, 
                          html = TRUE),
            pos = colDef(name = "Position", maxWidth = 70, 
                         style = list(fontWeight = "bold")),
            snaps = colDef(name = "Total Snaps", maxWidth = 70, 
                           style = list(fontWeight = "bold")),
            team_logo_espn = colDef(name = "Team", maxWidth = 70,
                                    cell = embed_img(height = 20, width = 20)), 
            xT = colDef(name = "xT", maxWidth = 120, 
                              cell = color_tiles(t0,colors = viridis::plasma(10, direction = -1),
                                                 bold_text = TRUE,
                                                 box_shadow = TRUE)),
            xT.snap = colDef(name = "xT/Snap", maxWidth = 120, 
                               cell = color_tiles(t0,colors = viridis::plasma(10, direction = -1), 
                                                  bold_text = TRUE,
                                                  box_shadow = TRUE,
                                                  number_fmt = scales::number_format(accuracy = 0.01)))
            )
)

f0 <-"plots/tackle.html"
saveWidget(tackle, file.path(normalizePath(dirname(f0)),basename(f0)))


display_html('<center>
<div class="prs">
    <div class="title" style="text-align:left;">
        <h2 style="font-size:24px"><b>Tackling Metrics</b></h2>
        <i>2022 NFL Season: Weeks 1-10</i>
    </div>
    <iframe src="plots/tackle.html" align="center" width="100%" height="500" frameBorder="0"></iframe>
    <span style="font-style:italic;font-size:15px">While underrepresented by traditional tackling metrics, 
players like Maxx Crosby and Denzel Perryman score highly in expected tackles</span><br>
</div>
<center>')