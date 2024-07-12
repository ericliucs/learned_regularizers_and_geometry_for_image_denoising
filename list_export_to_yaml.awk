#!/usr/bin/env awk -f
#' This awk script was copied from a StackOverflow answer:
#' https://stackoverflow.com/questions/70774618/conda-create-from-requirements-txt-not-finding-packages
#'
#' Trying to create an environment using Ryan's requirements.txt does not seem to work when using
#' the suggested `conda create` command.  Apparently, the package listings ending in =pypi_0 indicate
#' that Ryan originally installed these in his environment using pip.  Whenever conda tries to find
#' these packages (using these strings) in its channels, it can't find them.
#'
#' A YAML file can also be used to create an environment.  This script converts a requirements.txt
#' file with broken pip-installed package references to a YAML file with listings that conda
#' will correctly find.
#'	- NK (03/21/24)
#'
#' Author: Mervin Fansler
#' GitHub: @mfansler
#' License: MIT
#' 
#' Basic usage
#' $ conda list --export | awk -f list_export_to_yaml.awk
#' 
#' Omitting builds with 'no_builds'
#' $ conda list --export | awk -v no_builds=1 -f list_export_to_yaml.awk
#' 
#' Specifying channels with 'channels'
#' $ conda list --export | awk -v channels="conda-forge,defaults" -f list_export_to_yaml.awk

BEGIN {
  FS="=";
  if (channels) split(channels, channels_arr, ",");
  else channels_arr[0]="defaults";
}
{
  # skip header
  if ($1 ~ /^#/) next;

  if ($3 ~ /pypi/) {  # pypi packages
    pip=1;
    pypi[i++]="    - "$1"=="$2" ";
  } else {  # conda packages
    if ($1 ~ /pip/) pip=1;
    else {  # should we keep builds?
      if (no_builds) conda[j++]="  - "$1"="$2" ";
      else conda[j++]="  - "$1"="$2"="$3" ";
    }
  }
}
END {
  # emit channel info
  print "channels: ";
  for (k in channels_arr) print "  - "channels_arr[k]" ";

  # emit conda pkg info
  print "dependencies: ";
  for (j in conda) print conda[j];

  # emit PyPI pkg info
  if (pip) print "  - pip ";
  if (length(pypi) > 0) {
      print "  - pip: ";
      for (i in pypi) print pypi[i];
  }
}


