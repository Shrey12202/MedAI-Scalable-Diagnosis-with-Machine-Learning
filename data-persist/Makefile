all: \
	index.md \
	0_intro.ipynb \
	1_create_server.ipynb \
	2_object.ipynb \
	3_block.ipynb \
	4_delete.ipynb

clean: 
	rm index.md \
	0_intro.ipynb \
	1_create_server.ipynb \
	2_object.ipynb \
	3_block.ipynb \
	4_delete.ipynb

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/create_server.md \
		snippets/object.md \
		snippets/block.md \
		snippets/delete.md \
		snippets/footer.md \
		> index.tmp.md
	grep -v '^:::' index.tmp.md > index.md
	rm index.tmp.md
	cat snippets/footer.md >> index.md

0_intro.ipynb: snippets/intro.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/intro.md \
                -o 0_intro.ipynb  
	sed -i 's/attachment://g' 0_intro.ipynb


1_create_server.ipynb: snippets/create_server.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_server.md \
                -o 1_create_server.ipynb  
	sed -i 's/attachment://g' 1_create_server.ipynb

2_object.ipynb: snippets/object.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/object.md \
				-o 2_object.ipynb  
	sed -i 's/attachment://g' 2_object.ipynb

3_block.ipynb: snippets/block.md	
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/block.md \
				-o 3_block.ipynb  
	sed -i 's/attachment://g' 3_block.ipynb

4_delete.ipynb: snippets/delete.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/delete.md \
				-o 4_delete.ipynb  
	sed -i 's/attachment://g' 4_delete.ipynb