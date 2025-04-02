library(dplyr)
library(readr)
library(igraph)

#' Constructs a subnetwork of MIPPIE
#'
#' Given a list of proteins or interactions of interest, it constructs a
#' subnetwork around them using a user-specified neighborhood order.
#'
#' @param query.file character; The path to the file containing the proteins or
#' PPIs of interest. Protein pairs must be tab-separated.
#' @param path.to.mippie character; The path to the file containing MIPPIE
#' database.
#' @param path.to.proteins character; The path to the file containing protein
#' meta-data.
#' @param order integer; The order of the neighborhood, default value is 1.
#' @param output.file character; Path and name of the output files, which will
#' contain the constructed subnetwork around the input proteins.
#'
#' @return An igraph representation of the constructed subnetwork. In addition,
#' it writes a tibble to the file indicated by parameter output.file containing
#' the edges of the subnetwork.
#'
#' @author Gregorio Alanis-Lobato \email{galanisl@uni-mainz.de}
#'
#' @examples
#'
#' mippie_subnet <- mippie_nc(query.file = "query.txt",
#'                            path.to.mippie = "mippie_ppi_v1_0.tsv",
#'                            path.to.proteins = "mippie_proteins_v1_0.tsv",
#'                            order = 2, output.file = "mippie_subnet.tsv")
#'
mippie_nc <- function(query.file, path.to.mippie = "mippie_ppi_v1_0.tsv",
                      path.to.proteins = "mippie_proteins_v1_0.tsv",
                      order = 1, output.file = "mippie_subnet.tsv"){
  # Read in the query file
  qry <- scan(query.file, sep = "\n", what = "character", quiet = TRUE)
  qry <- strsplit(qry, split = "\t")
  qry_idx <- unique(unlist(qry))

  # Read in the edge and node data
  vtx <- read_tsv(path.to.proteins)
  edg <- read_tsv(path.to.mippie)

  # Build an igraph object with the edge and node data
  g <- graph_from_data_frame(d = edg, directed = FALSE, vertices = vtx)

  # Find the query IDs in the graph
#   idx <- integer(length = length(qry_idx))
  idx <- list()
  x <- 1
  for(i in 1:length(qry_idx)){
    a <- get_node_idx(g, qry_idx[i])
    if(a != -1){
        idx[x] <- a
        x <- x + 1
    }
#     if(idx[i] == -1){
#       stop(paste0(qry_idx[i], " is not a valid protein ID!!! ",
#                   "Subnetwork couldn't be constructed."))
#     }
  }

  # Build a subnetwork around the given nodes
  subnet <- get_orderX_net(g, idx, order = order)

  # Check if the given edges are in the network, if not add them with score -1
  ppi_idx <- which(lengths(qry) > 1)
  for(ppi in qry[ppi_idx]){
    pA <- get_node_idx(subnet, ppi[1])
    pB <- get_node_idx(subnet, ppi[2])
    if(!are_adjacent(subnet, pA, pB)){
      subnet <- add_edges(subnet, c(pA, pB), MIPPIE_score = -1)
    }
  }

  #Write subnetwork to output file
  sn_edg <- igraph::as_data_frame(subnet, what = "edges") %>%
    select(from, to, MIPPIE_score) %>%
    rename(entrezA = from, entrezB = to)
  write_tsv(sn_edg, path = output.file)

  return(subnet)
}


#' Finds the numeric index of a node
#'
#' Given an igraph object and a vertex ID, it finds its numeric index in the
#' network
#'
#' @param net igraph; The reference network.
#' @param vtx character; The node ID.
#'
#' @return An integer with the numeric ID of the node in the igraph object.
#'
get_node_idx <- function(net, vtx){
  if(tolower(vtx) %in% tolower(V(net)$name)){
    idx <- which(tolower(V(net)$name) == tolower(vtx))
  }else if(tolower(vtx) %in% tolower(V(net)$official_symbol)){
    idx <- which(tolower(V(net)$official_symbol) == tolower(vtx))
  }else if(tolower(vtx) %in% tolower(V(net)$uniprot_accession)){
    idx <- which(tolower(V(net)$uniprot_accession) == tolower(vtx))
  }else if(tolower(vtx) %in% tolower(V(net)$mgi)){
    idx <- which(tolower(V(net)$mgi) == tolower(vtx))
  }else{
    idx <- -1
  }
  return(idx)
}

#' Induces a subgraph around the given nodes
#'
#' Given an igraph object and a vector of vertex IDs, it constructs a subnetwork
#' around them using a the specified neighborhood order.
#'
#' @param net igraph; The reference network.
#' @param vtx integer; The node IDs.
#' @param order integer; The order of the neighborhood, default value is 1.
#'
#' @return An igraph object with the constructed subnetwork.
#'
get_orderX_net <- function(net, vtx, order = 1){
  idx <- ego(net, order = order, nodes = vtx)
  idx <- unique(unlist(idx))
  return(induced_subgraph(net, idx))
}

args <- commandArgs(trailingOnly=TRUE)
dataset <- args[1]
output <- args[2]
mippie_subnet <- mippie_nc(query.file = dataset,
path.to.mippie = "mippie_ppi_v1_0.tsv",
path.to.proteins = "mippie_proteins_v1_0.tsv", order = 1, output.file = output);
