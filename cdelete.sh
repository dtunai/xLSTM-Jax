git checkout --orphan TEMP_BRANCH
git add -A
git commit -am "Initial commit"
git branch -D main
git branch -m main
git push -f origin main