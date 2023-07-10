#!/bin/sh
echo "hi from poetry "


git config --global user.email "h.javedani@gmail.com"
git config --global user.name $gitusername
git config --global user.password $gitpassword


nox -s release_lohrasb -- minor $gitusername 'h.javedani@gmail.com' $gitpassword

poetry build
poetry publish --username=$username --password=$password

