all: \
	index.md \
	0_intro.ipynb \
	1_create_server.ipynb \
	2_save_production_data.ipynb \
	3_setup_labelstudio.ipynb \
	workspace/4_eval_production.ipynb \
	workspace/5_random_sample.ipynb \
	workspace/6_low_confidence_sample.ipynb \
	workspace/7_explicit_user_feedback.ipynb \
	8_scheduling.ipynb \
	9_delete.ipynb

clean: 
	rm index.md \
	0_intro.ipynb \
	1_create_server.ipynb \
	2_save_production_data.ipynb \
	3_setup_labelstudio.ipynb \
	workspace/4_eval_production.ipynb \
	workspace/5_random_sample.ipynb \
	workspace/6_low_confidence_sample.ipynb \
	workspace/7_explicit_user_feedback.ipynb \
	8_scheduling.ipynb \
	9_delete.ipynb

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/create_server.md \
		snippets/save_production_data.md \
		snippets/setup_labelstudio.md \
		snippets/eval_production.md \
		snippets/random_sample.md \
		snippets/low_confidence_sample.md \
		snippets/explicit_user_feedback.md \
		snippets/scheduling.md \
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

2_save_production_data.ipynb: snippets/save_production_data.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/save_production_data.md \
				-o 2_save_production_data.ipynb  
	sed -i 's/attachment://g' 2_save_production_data.ipynb

3_setup_labelstudio.ipynb: snippets/setup_labelstudio.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/setup_labelstudio.md \
				-o 3_setup_labelstudio.ipynb  
	sed -i 's/attachment://g' 3_setup_labelstudio.ipynb

workspace/4_eval_production.ipynb: snippets/eval_production.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/eval_production.md \
				-o workspace/4_eval_production.ipynb  
	sed -i 's/attachment://g' workspace/4_eval_production.ipynb

workspace/5_random_sample.ipynb: snippets/random_sample.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/random_sample.md \
				-o workspace/5_random_sample.ipynb  
	sed -i 's/attachment://g' workspace/5_random_sample.ipynb

workspace/6_low_confidence_sample.ipynb: snippets/low_confidence_sample.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/low_confidence_sample.md \
				-o workspace/6_low_confidence_sample.ipynb  
	sed -i 's/attachment://g' workspace/6_low_confidence_sample.ipynb

workspace/7_explicit_user_feedback.ipynb: snippets/explicit_user_feedback.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/explicit_user_feedback.md \
				-o workspace/7_explicit_user_feedback.ipynb  
	sed -i 's/attachment://g' workspace/7_explicit_user_feedback.ipynb

8_scheduling.ipynb: snippets/scheduling.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/scheduling.md \
				-o 8_scheduling.ipynb  
	sed -i 's/attachment://g' 8_scheduling.ipynb

9_delete.ipynb: snippets/delete.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/delete.md \
				-o 9_delete.ipynb  
	sed -i 's/attachment://g' 9_delete.ipynb