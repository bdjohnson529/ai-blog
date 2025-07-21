source "https://rubygems.org"

ruby "3.3.4"

# Jekyll
# gem "jekyll", "~> 4.3"
gem "github-pages", group: :jekyll_plugins

# Jekyll plugins
group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-sitemap"
  gem "jekyll-seo-tag"
  gem "jekyll-paginate"
  gem "jekyll-gist"
#bundle update github-pages  gem "jekyll-include-cache"
end

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds since newer versions of the gem
# do not have a Java counterpart.
gem "http_parser.rb", :platforms => [:jruby]

# Webrick for Ruby 3.0+
gem "webrick"

# Rouge for syntax highlighting
gem "rouge"

# Kramdown for markdown processing
# gem "kramdown", "~> 2.4"
# gem "kramdown-parser-gfm", "~> 1.1"

# Faraday retry middleware
# gem "faraday-retry"

# For development
# group :development do
  # gem "jekyll-admin"
#end
