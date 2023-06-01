let $ = jQuery;

jQuery(document).ready(function() {

	// Determine localization.
	function getLanguage(thisLanguage) {

		// Catch TW.
		if ('zh-hant' === thisLanguage) {
			thisLanguage = 'zh_TW';
		}

		let newLanguage = 'en';
		let supportedLanguages = [
			'af',       // Afrikaans
			'sq',       // Albanian
			'ar',       // Arabic
			'az',       // Azerbaijani
			'eu',       // Basque
			'bs',       // Bosnian
			'bg',       // Bulgarian
			'ca',       // Catalan
			'zh',       // Chinese
			'zh_TW',    // Chinese - Taiwan
			'cs',       // Czech
			'da',       // Danish
			'nl',       // Dutch
			'en',       // English
			'et',       // Estonian
			'fi',       // Finnish
			'fr',       // French
			'fr_CA',    // French - Canada
			'de',       // German
			'el',       // Greek
			'he',       // Hebrew
			'hu',       // Hungarian
			'id',       // Indonesian
			'it',       // Italian
			'ja',       // Japanese
			'pam',      // Kapampangan
			'kk',       // Kazakh
			'km',       // Khmer
			'ko',       // Korean
			'lad',      // Ladino
			'lv',       // Latvian
			'lt',       // Lithuanian
			'mk',       // Macedonian
			'no',       // Norwegian
			'nn',       // Norwegian - Nyorsk
			'fa',       // Persian
			'pl',       // Polish
			'pt',       // Portugese
			'pt_BR',    // Portugese - Brazil
			'ro',       // Romanian
			'ru',       // Russian
			'sc',       // Sardinian
			'sr',       // Serbian
			'sr@latin', // Serbian - Latin
			'sk',       // Slovak
			'sl',       // Slovenian
			'es_AR',    // Spanish - Argentina
			'es_MX',    // Spanish - Mexico
			'es_ES',    // Spanish - Spain
			'sv',       // Swedish
			'th',       // Thai
			'tr',       // Turkish
			'uk',       // Ukrainian
			'ur',       // Urdu
			'vec',      // Venetian
			'vi',       // Vietnamese
			'cy',       // Welsh
		];
		if (supportedLanguages.indexOf(thisLanguage) > -1) {
			newLanguage = thisLanguage;
		}
		return newLanguage;
	}

	// For subsequent posts on the page, show "Load Comments" button.
	function setupLoadComments(thisLanguage) {

		jQuery('.cf-load-comments').off('click.cf_load_comments').on('click.cf_load_comments', function() {
			let $btn = $(this);
			let article = $btn.parents('article')[0];
			let $article = $(article);
			let identifier = $article.data('identifier');
			let permalink = $article.data('url');
			let title = $article.data('title');
			let disqusEl = $('#disqus_thread');
			if (identifier.length && permalink.length && title.length && disqusEl.length) {
				disqusEl.remove();
				$('<div id="disqus_thread"></div>').insertAfter($btn);
				loadDisqus(identifier, permalink, title, thisLanguage);
			}
		});
	}

	// Initialize the Disqus comment box.
	function initLoadDisqusComments() {

		let localizedLanguage = getLanguage(nvb4ThemeVars.language);
		let disqus_shortname = 'nvidiablog';

		if ('ru' === localizedLanguage) {
			disqus_shortname = 'ru-nvidia';
		}

		let disqus_config = function() {
			this.language = localizedLanguage;
			this.page.url = nvb4ThemeVars.disqus_url;
		};

		(function() {

			if (-1 === window.location.href.indexOf("reader-no-image")) {

				if (null === document.getElementById('disqus_thread')) {
					let dsqDiv = document.createElement('div');
					dsqDiv.setAttribute('id', 'disqus_thread');
					(document.getElementsByTagName('body')[0]).appendChild(dsqDiv);
				}
				let dsq = document.createElement('script');
				dsq.type = 'text/javascript';
				dsq.async = true;
				dsq.src = 'https://' + disqus_shortname + '.disqus.com/embed.js';
				(document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(dsq);
			}
		})();

		setupLoadComments(this.language);
	}

	// Reset Disqus.
	function loadDisqus(newIdentifier, newUrl, newTitle, thisLanguage) {

		let localizedLanguage = getLanguage(thisLanguage);

		DISQUS.reset({
			reload: true,
			config: function() {
				this.page.identifier = newIdentifier;
				this.page.url = newUrl;
				this.page.title = newTitle;
				this.language = localizedLanguage;
			}
		});
	}

	// Initialize.
	initLoadDisqusComments();

	// New elements.
	$(document).on('yith_infs_added_elem', function() {
		setupLoadComments(nvb4ThemeVars.language);
	});

});
